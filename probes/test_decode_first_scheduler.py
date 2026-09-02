#!/usr/bin/env python3
"""CPU-only unit checks for the decode-first scheduler adapter."""

from types import SimpleNamespace
from unittest.mock import patch

from glm_decode_first_scheduler import (
    AdaptiveDecodeFirstScheduler,
    DecodeFirstScheduler,
    _adaptive_credit_increment,
    _effective_packet_tokens,
    _select_tier,
    _validate_adaptive_profile,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler


def check(interval: int, current_step: int, incoming: bool, expected: bool) -> None:
    scheduler = DecodeFirstScheduler.__new__(DecodeFirstScheduler)
    scheduler.scheduler_config = SimpleNamespace(
        prefill_schedule_interval=interval,
        long_prefill_token_threshold=0,
    )
    scheduler.current_step = current_step
    scheduler.running = ()
    scheduler.prefill_capacity_bound = False

    sentinel = object()
    with patch.object(Scheduler, "schedule", return_value=sentinel) as parent:
        result = scheduler.schedule(throttle_prefills=incoming)

    assert result is sentinel
    parent.assert_called_once_with(throttle_prefills=expected)


def check_threshold_scope(
    running_is_prefill: tuple[bool, ...], expected_during_schedule: int
) -> None:
    scheduler = DecodeFirstScheduler.__new__(DecodeFirstScheduler)
    scheduler.scheduler_config = SimpleNamespace(
        prefill_schedule_interval=4,
        long_prefill_token_threshold=512,
    )
    scheduler.current_step = 0
    scheduler.running = tuple(
        SimpleNamespace(is_prefill_chunk=is_prefill)
        for is_prefill in running_is_prefill
    )
    scheduler.prefill_capacity_bound = False

    sentinel = object()

    def parent(*, throttle_prefills: bool):
        assert not throttle_prefills
        assert (
            scheduler.scheduler_config.long_prefill_token_threshold
            == expected_during_schedule
        )
        return sentinel

    with patch.object(Scheduler, "schedule", side_effect=parent):
        result = scheduler.schedule()

    assert result is sentinel
    assert scheduler.scheduler_config.long_prefill_token_threshold == 512


def check_capacity_bound_scope(current_step: int, expected_throttle: bool) -> None:
    scheduler = DecodeFirstScheduler.__new__(DecodeFirstScheduler)
    scheduler.scheduler_config = SimpleNamespace(
        prefill_schedule_interval=4,
        long_prefill_token_threshold=0,
    )
    scheduler.current_step = current_step
    scheduler.running = (SimpleNamespace(is_prefill_chunk=False),)
    scheduler.prefill_capacity_bound = True

    sentinel = object()

    def parent(*, throttle_prefills: bool):
        assert throttle_prefills is expected_throttle
        assert scheduler.prefill_capacity_bound is (not expected_throttle)
        return sentinel

    with patch.object(Scheduler, "schedule", side_effect=parent):
        result = scheduler.schedule()

    assert result is sentinel
    assert scheduler.prefill_capacity_bound is True


def adaptive_scheduler(
    num_decoders: int,
    num_prefills: int,
    *,
    global_budget: bool = True,
) -> AdaptiveDecodeFirstScheduler:
    scheduler = AdaptiveDecodeFirstScheduler.__new__(
        AdaptiveDecodeFirstScheduler
    )
    scheduler.scheduler_config = SimpleNamespace(
        prefill_schedule_interval=4,
        long_prefill_token_threshold=512,
    )
    scheduler.current_step = 0
    scheduler.running = [
        SimpleNamespace(request_id=f"d{index}", is_prefill_chunk=False)
        for index in range(num_decoders)
    ] + [
        SimpleNamespace(request_id=f"p{index}", is_prefill_chunk=True)
        for index in range(num_prefills)
    ]
    scheduler.waiting = ()
    scheduler.skipped_waiting = ()
    scheduler.max_num_running_reqs = 6
    scheduler.prefill_capacity_bound = False
    scheduler._adaptive_profile_cache = {
        "profile_name": "test",
        "global_budget": global_budget,
        "pure_prefill_token_threshold": 0,
        "tiers": [
            {"max_decoders": 2, "packet_tokens": 256, "interval": 2},
            {"max_decoders": 4, "packet_tokens": 384, "interval": 3},
            {"max_decoders": 6, "packet_tokens": 512, "interval": 2},
        ],
    }
    scheduler._adaptive_profile_next_check = float("inf")
    return scheduler


def adaptive_call(
    scheduler: AdaptiveDecodeFirstScheduler,
    *,
    expected_throttle: bool,
    expected_threshold: int,
    scheduled_prefill_tokens: tuple[int, ...] = (),
    raises: bool = False,
) -> None:
    scheduled = {
        f"p{index}": count
        for index, count in enumerate(scheduled_prefill_tokens)
    }
    scheduled.update(
        {
            request.request_id: 1
            for request in scheduler.running
            if not request.is_prefill_chunk
        }
    )

    def parent(*, throttle_prefills: bool):
        assert throttle_prefills is expected_throttle
        assert (
            scheduler.scheduler_config.long_prefill_token_threshold
            == expected_threshold
        )
        if raises:
            raise RuntimeError("sentinel")
        return SimpleNamespace(num_scheduled_tokens=scheduled)

    with patch.object(Scheduler, "schedule", side_effect=parent):
        if raises:
            try:
                scheduler.schedule()
            except RuntimeError as exc:
                assert str(exc) == "sentinel"
            else:
                raise AssertionError("adaptive parent exception was swallowed")
        else:
            scheduler.schedule()
    assert scheduler.scheduler_config.long_prefill_token_threshold == 512


def check_adaptive_scheduler() -> None:
    assert issubclass(AdaptiveDecodeFirstScheduler, AsyncScheduler)
    assert _adaptive_credit_increment(256, 2) == 128
    assert _adaptive_credit_increment(384, 3) == 128
    assert _adaptive_credit_increment(512, 2) == 256

    profile = _validate_adaptive_profile({}, 512, 4)
    assert _select_tier(profile, 1)["interval"] == 4
    assert _select_tier(profile, 3)["interval"] == 3
    assert _select_tier(profile, 5)["interval"] == 2
    scaled_profile = _validate_adaptive_profile(
        {
            "tiers": [
                {
                    "max_decoders": 6,
                    "packet_tokens": 256,
                    "interval": 4,
                    "packet_tokens_per_extra_prefill": 64,
                    "max_packet_tokens": 384,
                }
            ]
        },
        512,
        4,
    )
    scaled_tier = scaled_profile["tiers"][0]
    assert _effective_packet_tokens(scaled_tier, 1) == 256
    assert _effective_packet_tokens(scaled_tier, 2) == 320
    assert _effective_packet_tokens(scaled_tier, 3) == 384
    assert _effective_packet_tokens(scaled_tier, 5) == 384

    # C1/C2: an immediate 256-token packet, one protected step, then another
    # packet. This has the same average allowance as static 512/4.
    low = adaptive_scheduler(1, 1)
    adaptive_call(
        low,
        expected_throttle=False,
        expected_threshold=256,
        scheduled_prefill_tokens=(256,),
    )
    assert low._mixed_prefill_credit == 0
    adaptive_call(low, expected_throttle=True, expected_threshold=256)
    assert low._mixed_prefill_credit == 128
    adaptive_call(
        low,
        expected_throttle=False,
        expected_threshold=256,
        scheduled_prefill_tokens=(256,),
    )
    assert low._mixed_prefill_credit == 0

    # C3/C4 earn a 384-token packet over three steps.
    middle = adaptive_scheduler(3, 1)
    adaptive_call(
        middle,
        expected_throttle=False,
        expected_threshold=384,
        scheduled_prefill_tokens=(384,),
    )
    adaptive_call(middle, expected_throttle=True, expected_threshold=384)
    adaptive_call(middle, expected_throttle=True, expected_threshold=384)
    adaptive_call(
        middle,
        expected_throttle=False,
        expected_threshold=384,
        scheduled_prefill_tokens=(384,),
    )

    # C5 earns its packet every two steps.
    high = adaptive_scheduler(5, 1)
    adaptive_call(
        high,
        expected_throttle=False,
        expected_threshold=512,
        scheduled_prefill_tokens=(512,),
    )
    adaptive_call(high, expected_throttle=True, expected_threshold=512)
    adaptive_call(
        high,
        expected_throttle=False,
        expected_threshold=512,
        scheduled_prefill_tokens=(512,),
    )

    # Two prefills share one strict global packet.
    shared = adaptive_scheduler(1, 2)
    adaptive_call(
        shared,
        expected_throttle=False,
        expected_threshold=128,
        scheduled_prefill_tokens=(128, 128),
    )
    assert shared._mixed_prefill_credit == 0

    # The optional extra-prefill allowance grows the aggregate packet, then
    # the existing global budget divides it evenly across runnable prefills.
    scaled = adaptive_scheduler(3, 3)
    scaled._adaptive_profile_cache["tiers"][1].update(
        packet_tokens=256,
        interval=4,
        packet_tokens_per_extra_prefill=64,
        max_packet_tokens=384,
    )
    adaptive_call(
        scaled,
        expected_throttle=False,
        expected_threshold=128,
        scheduled_prefill_tokens=(128, 128, 128),
    )
    assert scaled._mixed_prefill_credit == 0

    # Pure prefill remains uncapped, and parent failures restore configuration.
    pure = adaptive_scheduler(0, 1)
    adaptive_call(pure, expected_throttle=False, expected_threshold=0)
    pure.running = [
        *(
            SimpleNamespace(request_id=f"d{index}", is_prefill_chunk=False)
            for index in range(3)
        ),
        *(
            SimpleNamespace(request_id=f"p{index}", is_prefill_chunk=True)
            for index in range(3)
        ),
    ]
    pure._adaptive_profile_cache["tiers"][1].update(
        packet_tokens=256,
        interval=4,
        packet_tokens_per_extra_prefill=64,
        max_packet_tokens=384,
    )
    adaptive_call(
        pure,
        expected_throttle=False,
        expected_threshold=128,
        scheduled_prefill_tokens=(128, 128, 128),
    )
    assert pure._mixed_prefill_credit == 0
    failing = adaptive_scheduler(1, 1)
    adaptive_call(
        failing,
        expected_throttle=False,
        expected_threshold=256,
        raises=True,
    )


def main() -> None:
    assert issubclass(DecodeFirstScheduler, AsyncScheduler)

    # Interval one is stock behavior. Higher intervals admit step zero and
    # then defer until the next cadence boundary. An upstream throttle signal
    # (for example from a DP core) must never be discarded.
    cases = (
        (1, 0, False, False),
        (1, 9, False, False),
        (4, 0, False, False),
        (4, 1, False, True),
        (4, 3, False, True),
        (4, 4, False, False),
        (4, 4, True, True),
    )
    for case in cases:
        check(*case)

    check_threshold_scope((), 0)
    check_threshold_scope((True, True), 0)
    check_threshold_scope((False, True), 512)
    check_capacity_bound_scope(1, True)
    check_capacity_bound_scope(4, False)
    check_adaptive_scheduler()
    print(
        f"decode-first scheduler: {len(cases)} cadence checks, "
        "3 threshold-scope checks, 2 saturation checks, and adaptive "
        "credit/profile checks passed"
    )


if __name__ == "__main__":
    main()
