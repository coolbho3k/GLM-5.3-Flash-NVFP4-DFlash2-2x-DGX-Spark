# Adaptive scheduler profiles

The active adaptive profile is adaptive.json. When the server is launched
with PREFILL_ADMISSION_POLICY=adaptive, the complete directory is mounted
read-only into the container and the scheduler checks this file for changes at
most once per second. Editing it therefore does not require a model reload.

Each tier applies up to its inclusive max_decoders value:

- packet_tokens is the approximate aggregate mixed-prefill budget.
- interval is the number of decode-bearing engine steps needed to earn one
  full packet.
- packet_tokens_per_extra_prefill optionally grows the aggregate packet for
  each runnable prefill after the first. This recovers throughput only when
  several new sessions compete, rather than penalizing the common one-prefill
  case.
- max_packet_tokens caps that growth. It defaults to packet_tokens, so the
  feature is opt-in per tier.
- global_budget=true divides a packet among runnable prefills.
- pure_prefill_token_threshold=0 leaves pure prefill uncapped.

Only edit the profile while no benchmark wave is active. The scheduler
validates every reload and retains the last valid profile if an update is
malformed.
