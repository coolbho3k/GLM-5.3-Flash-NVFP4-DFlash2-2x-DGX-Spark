#!/usr/bin/env python3
"""CPU-only unit checks for the decode-first scheduler adapter."""

from types import SimpleNamespace
from unittest.mock import patch

from glm_decode_first_scheduler import DecodeFirstScheduler
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
    print(
        f"decode-first scheduler: {len(cases)} cadence checks, "
        "3 threshold-scope checks, and 2 saturation checks passed"
    )


if __name__ == "__main__":
    main()
