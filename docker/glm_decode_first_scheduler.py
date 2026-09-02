"""Decode-first cadence adapters for the stock vLLM V1 scheduler.

The vLLM scheduler in the GLM serving image already knows how to defer
prefills while decoders are active. The ordinary (non-data-parallel)
EngineCore never enables that path, however. This adapter makes the existing
``prefill_schedule_interval`` setting effective without replacing or copying
the scheduler implementation. It must inherit AsyncScheduler: this stack uses
its placeholder lifecycle to keep DFlash speculative decoding active even at
single-request concurrency.
"""

import json
import logging
import os
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


logger = logging.getLogger(__name__)


def _adaptive_credit_increment(packet_tokens: int, interval: int) -> int:
    """Return mixed-prefill credit earned by one decode-bearing step."""
    if packet_tokens <= 0:
        return 0
    interval = max(1, interval)
    return (packet_tokens + interval - 1) // interval


def _validate_adaptive_profile(
    raw: dict, fallback_packet: int, fallback_interval: int
) -> dict:
    """Validate and normalize a hot-reloadable adaptive profile."""
    default_tiers = [
        {
            "max_decoders": 2,
            "packet_tokens": fallback_packet,
            "interval": fallback_interval,
        },
        {
            "max_decoders": 4,
            "packet_tokens": fallback_packet,
            "interval": max(1, fallback_interval - 1),
        },
        {
            "max_decoders": 6,
            "packet_tokens": fallback_packet,
            "interval": max(1, fallback_interval - 2),
        },
    ]
    tiers = raw.get("tiers", default_tiers)
    if not isinstance(tiers, list) or not tiers:
        raise ValueError("adaptive scheduler tiers must be a non-empty list")
    normalized_tiers = []
    previous_max = 0
    for tier in tiers:
        if not isinstance(tier, dict):
            raise ValueError("each adaptive scheduler tier must be an object")
        max_decoders = int(tier["max_decoders"])
        packet_tokens = int(tier["packet_tokens"])
        interval = int(tier["interval"])
        extra_prefill_tokens = int(
            tier.get("packet_tokens_per_extra_prefill", 0)
        )
        max_packet_tokens = int(
            tier.get("max_packet_tokens", packet_tokens)
        )
        if max_decoders <= previous_max:
            raise ValueError("tier max_decoders values must be strictly increasing")
        if packet_tokens <= 0 or interval <= 0:
            raise ValueError("tier packet_tokens and interval must be positive")
        if extra_prefill_tokens < 0:
            raise ValueError(
                "tier packet_tokens_per_extra_prefill must be non-negative"
            )
        if max_packet_tokens < packet_tokens:
            raise ValueError(
                "tier max_packet_tokens must be at least packet_tokens"
            )
        normalized_tiers.append(
            {
                "max_decoders": max_decoders,
                "packet_tokens": packet_tokens,
                "interval": interval,
                "packet_tokens_per_extra_prefill": extra_prefill_tokens,
                "max_packet_tokens": max_packet_tokens,
            }
        )
        previous_max = max_decoders
    if previous_max < 6:
        raise ValueError("adaptive scheduler tiers must cover six decoders")

    global_budget = raw.get("global_budget", True)
    if not isinstance(global_budget, bool):
        raise ValueError("global_budget must be true or false")
    pure_threshold = int(raw.get("pure_prefill_token_threshold", 0))
    if pure_threshold < 0:
        raise ValueError("pure_prefill_token_threshold must be non-negative")
    return {
        "profile_name": str(raw.get("profile_name", "adaptive")),
        "global_budget": global_budget,
        "pure_prefill_token_threshold": pure_threshold,
        "tiers": normalized_tiers,
    }


def _adaptive_profile(scheduler, fallback_packet: int, fallback_interval: int) -> dict:
    """Return the current profile, reloading a mounted JSON file at most once/s."""
    now = time.monotonic()
    cached = getattr(scheduler, "_adaptive_profile_cache", None)
    next_check = getattr(scheduler, "_adaptive_profile_next_check", 0.0)
    if cached is not None and now < next_check:
        return cached
    scheduler._adaptive_profile_next_check = now + float(
        os.getenv("GLM_ADAPTIVE_PREFILL_RELOAD_SECONDS", "1")
    )

    path = os.getenv("GLM_ADAPTIVE_PREFILL_CONFIG", "")
    if not path:
        profile = _validate_adaptive_profile(
            {}, fallback_packet, fallback_interval
        )
        scheduler._adaptive_profile_cache = profile
        return profile

    try:
        mtime_ns = os.stat(path).st_mtime_ns
        if (
            cached is not None
            and mtime_ns == getattr(scheduler, "_adaptive_profile_mtime_ns", None)
        ):
            return cached
        with open(path, encoding="utf-8") as profile_file:
            profile = _validate_adaptive_profile(
                json.load(profile_file), fallback_packet, fallback_interval
            )
        scheduler._adaptive_profile_cache = profile
        scheduler._adaptive_profile_mtime_ns = mtime_ns
        logger.info(
            "Loaded adaptive prefill profile %s: %s",
            profile["profile_name"],
            profile,
        )
        return profile
    except Exception:
        if cached is None:
            raise
        logger.exception("Ignoring invalid adaptive prefill profile update")
        return cached


def _select_tier(profile: dict, num_decoders: int) -> dict:
    for tier in profile["tiers"]:
        if num_decoders <= tier["max_decoders"]:
            return tier
    return profile["tiers"][-1]


def _effective_packet_tokens(tier: dict, num_prefills: int) -> int:
    """Scale aggregate work only when several prefills share the packet."""
    base = tier["packet_tokens"]
    extra = tier.get("packet_tokens_per_extra_prefill", 0)
    maximum = tier.get("max_packet_tokens", base)
    return min(maximum, base + max(0, num_prefills - 1) * extra)


def _schedulable_prefill_count(scheduler) -> int:
    """Approximate prefills that can run without exceeding sequence slots."""
    running_prefills = sum(
        request.is_prefill_chunk for request in scheduler.running
    )
    free_slots = max(
        0, scheduler.max_num_running_reqs - len(scheduler.running)
    )
    waiting_prefills = sum(
        request.is_prefill_chunk
        for queue in (scheduler.waiting, scheduler.skipped_waiting)
        for request in queue
    )
    return running_prefills + min(free_slots, waiting_prefills)


class DecodeFirstScheduler(AsyncScheduler):
    """Admit mixed-workload prefills once every configured engine steps."""

    def schedule(self, throttle_prefills: bool = False):
        interval = self.scheduler_config.prefill_schedule_interval
        cadence_throttles = interval > 1 and self.current_step % interval != 0
        should_throttle = throttle_prefills or cadence_throttles

        has_active_decoder = any(
            not request.is_prefill_chunk for request in self.running
        )

        # long_prefill_token_threshold is useful for bounding latency spikes in
        # a mixed step, but applying it to a pure-prefill batch needlessly
        # lowers throughput. Scheduler.schedule() is synchronous, so scoping
        # these mutable settings to the parent call is safe.
        threshold = self.scheduler_config.long_prefill_token_threshold
        uncap_pure_prefill = threshold > 0 and not has_active_decoder

        # Stock DP balancing releases the throttle while its waiting queue is
        # capacity-bound. Interactive cadence must stay strict under saturation
        # or queued prefills defeat the decode-first policy.
        capacity_bound = self.prefill_capacity_bound
        force_cadence = cadence_throttles and has_active_decoder and capacity_bound

        if uncap_pure_prefill:
            self.scheduler_config.long_prefill_token_threshold = 0
        if force_cadence:
            self.prefill_capacity_bound = False
        try:
            return super().schedule(throttle_prefills=should_throttle)
        finally:
            if uncap_pure_prefill:
                self.scheduler_config.long_prefill_token_threshold = threshold
            if force_cadence:
                self.prefill_capacity_bound = capacity_bound


class AdaptiveDecodeFirstScheduler(AsyncScheduler):
    """Decode-first scheduler with concurrency-sensitive prefill credit.

    long_prefill_token_threshold is treated as one aggregate mixed-prefill
    packet. Stock vLLM only exposes a per-request threshold, so this adapter
    divides the packet across prefills that can currently occupy request slots.
    """

    def schedule(self, throttle_prefills: bool = False):
        configured_threshold = (
            self.scheduler_config.long_prefill_token_threshold
        )
        configured_interval = self.scheduler_config.prefill_schedule_interval
        num_decoders = sum(
            not request.is_prefill_chunk for request in self.running
        )
        num_prefills = _schedulable_prefill_count(self)
        has_active_decoder = num_decoders > 0
        profile = _adaptive_profile(
            self, configured_threshold, configured_interval
        )
        tier = _select_tier(profile, max(1, num_decoders))
        packet_tokens = _effective_packet_tokens(tier, num_prefills)
        interval = tier["interval"]
        mixed_active = has_active_decoder and num_prefills > 0
        was_mixed_active = getattr(self, "_mixed_prefill_active", False)

        credit = min(
            packet_tokens,
            getattr(self, "_mixed_prefill_credit", packet_tokens),
        )
        if not mixed_active:
            # Pure prefill is uncapped. Decode-only time fills the bucket so
            # the next arriving prefill can start immediately.
            credit = packet_tokens
            admit_packet = False
        elif not was_mixed_active:
            # A packet is immediately available after idle/decode-only time,
            # including when several new prefills grow the effective packet.
            credit = packet_tokens
            admit_packet = not throttle_prefills
        else:
            credit = min(
                packet_tokens,
                credit + _adaptive_credit_increment(packet_tokens, interval),
            )
            admit_packet = credit >= packet_tokens and not throttle_prefills

        should_throttle = throttle_prefills or (
            has_active_decoder and num_prefills > 0 and not admit_packet
        )
        scoped_threshold = packet_tokens
        if not has_active_decoder:
            scoped_threshold = profile["pure_prefill_token_threshold"]
        elif admit_packet:
            # Floor division makes this a strict aggregate upper bound. At
            # MAX_NUM_SEQS=6, rounding can leave at most five tokens unused.
            if profile["global_budget"]:
                scoped_threshold = max(1, packet_tokens // num_prefills)

        # Stock DP balancing can release a throttle when its waiting queue is
        # capacity-bound. Interactive protection remains authoritative.
        capacity_bound = self.prefill_capacity_bound
        force_throttle = (
            should_throttle and has_active_decoder and capacity_bound
        )

        prefill_ids = {
            request.request_id
            for queue in (self.running, self.waiting, self.skipped_waiting)
            for request in queue
            if request.is_prefill_chunk
        }

        self.scheduler_config.long_prefill_token_threshold = scoped_threshold
        if force_throttle:
            self.prefill_capacity_bound = False
        try:
            output = super().schedule(throttle_prefills=should_throttle)
            if admit_packet:
                scheduled_prefill_tokens = sum(
                    count
                    for request_id, count in output.num_scheduled_tokens.items()
                    if request_id in prefill_ids
                )
                credit = max(0, credit - scheduled_prefill_tokens)
            self._mixed_prefill_credit = credit
            self._mixed_prefill_active = mixed_active
            return output
        finally:
            self.scheduler_config.long_prefill_token_threshold = (
                configured_threshold
            )
            if force_throttle:
                self.prefill_capacity_bound = capacity_bound
