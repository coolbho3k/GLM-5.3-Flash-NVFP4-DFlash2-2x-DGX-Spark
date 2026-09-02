"""Decode-first cadence adapter for the stock vLLM V1 scheduler.

The vLLM scheduler in the GLM serving image already knows how to defer
prefills while decoders are active. The ordinary (non-data-parallel)
EngineCore never enables that path, however. This adapter makes the existing
``prefill_schedule_interval`` setting effective without replacing or copying
the scheduler implementation. It must inherit AsyncScheduler: this stack uses
its placeholder lifecycle to keep DFlash speculative decoding active even at
single-request concurrency.
"""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


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
