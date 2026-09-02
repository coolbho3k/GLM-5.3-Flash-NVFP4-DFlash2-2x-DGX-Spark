# Adaptive Mixed Prefill/Decode Scheduler Plan

Status: implemented and benchmarked on 2026-09-02. The promoted profile and
measurements are in [ADAPTIVE-SCHEDULER-RESULTS.md](ADAPTIVE-SCHEDULER-RESULTS.md).
This document retains the original design and acceptance plan for provenance.

## Objective

Improve utilization and prefill progress when two to six interactive sessions
share the GLM-5.3-Flash server, while preserving responsive decode. Decode
latency is the primary objective; prefill TTFT and aggregate throughput are
secondary objectives.

This campaign must preserve:

- The validated 1M-context serving profile and its KV capacity.
- DFlash speculative decoding and the `AsyncScheduler` request lifecycle.
- NVFP4 weights, calibrated native-FP4 MLA KV, FP8 indexer cache, DCP2, and the
  current Marlin MoE backend unless a separate experiment explicitly changes
  one of them.
- Full 8,192-token scheduling capacity for pure-prefill workloads.
- The committed static `512 tokens / 4 steps` policy as an immediate rollback.

The prepared FlashInfer B12X experiment is separate. Do not combine a new MoE
backend and a new scheduler in the same measurement.

## Current behavior

The current `DecodeFirstScheduler` is a small adapter around vLLM's stock
`AsyncScheduler`:

- Running decode requests are scheduled first by vLLM.
- If any decoder is active, prefill work is admitted once every four engine
  steps.
- On an admitted step, `long_prefill_token_threshold=512` caps **each prefill
  request** at 512 tokens. It is not an aggregate mixed-prefill limit.
- When no decoder is active, the adapter removes that threshold and pure
  prefill may use the normal aggregate 8,192-token batch budget.
- `MAX_NUM_SEQS=6` limits the server to six running requests. If all six are
  decoders, an additional prefill cannot enter without increasing this limit
  or preempting a decoder.

The static policy is deliberately decode-biased. It protects decode, but it
does not react to concurrency and can leave most of the token budget unused on
decode-only steps. With several prefills, the per-request 512-token cap can
also produce a much larger mixed step than intended.

## Design principles

1. Always schedule eligible decode work before prefill work.
2. Control aggregate mixed-prefill work, not merely each request's chunk size.
3. Become more work-conserving as decode concurrency makes model execution
   better utilized.
4. Bound the number of consecutive decode-only steps so prefills cannot
   starve.
5. Share mixed-prefill capacity fairly without letting one long prompt create
   head-of-line blocking.
6. Keep pure-prefill execution fast and unchanged.
7. Make every new behavior opt-in and independently reversible.
8. Optimize from measurements of wall-clock TTFT and decode gaps, not scheduler
   iteration counts alone.

## Stage 1: plug-in-only adaptive scheduler

Add a second `AsyncScheduler` subclass and leave `DecodeFirstScheduler`
unchanged. Select the new class through a launch option such as
`PREFILL_ADMISSION_POLICY=adaptive`; retain `static` as the default until the
new policy wins the acceptance gates.

### Token-credit policy

Treat the configured 512-token threshold as one mixed-prefill packet. Prefill
credit accumulates on engine steps and is spent when a packet is admitted.
The initial conservative profile should be:

| Active decoders | Target mixed-prefill service | Rationale |
|---:|---:|---|
| 1-2 | approximately 512 tokens every 4 steps | Preserve C1/C2 interactivity. |
| 3-4 | approximately 512 tokens every 3 steps | Use capacity recovered from batching. |
| 5 | approximately 512 tokens every 2 steps | Keep queued/running prefill moving under pressure. |
| 6 | no additional request slot | All six sequence slots are occupied by decoders. |

The exact implementation should use accumulated integer credit rather than
`current_step % interval`. Credit makes transitions between concurrency levels
smooth and allows an unused packet to be available immediately when a prefill
arrives.

### Plug-in-only aggregate approximation

The stock scheduler exposes a per-request threshold, not a global prefill
budget. For the first implementation:

1. Count running prefills plus waiting prefills that could fit into currently
   available sequence slots.
2. Divide the 512-token packet among those candidates.
3. Temporarily install that fair-share value as
   `long_prefill_token_threshold` for the parent `schedule()` call.
4. Keep vLLM's existing decode-first ordering.
5. Restore all mutated scheduler configuration in `finally` blocks.
6. Disable vLLM's capacity-bound throttle escape on protected decode steps, as
   the current adapter already does.

This bounds aggregate work to approximately 512 tokens, with at most small
rounding loss. It will not redistribute the unused part of a share when a
prefill needs fewer tokens; that limitation is acceptable for the first A/B.

### Required safeguards

- Inherit from `AsyncScheduler`, never the plain `Scheduler`, because the plain
  scheduler previously disabled the working DFlash lifecycle and reduced
  decode to roughly 11 tok/s.
- An upstream `throttle_prefills=True` signal must never be overridden.
- A zero configured threshold should have explicit behavior: either use the
  existing static cadence without a packet cap or reject adaptive mode with a
  clear validation error. Do not silently create an unbounded adaptive packet.
- Pure-prefill mode must temporarily restore an uncapped threshold.
- If no prefill can occupy a sequence slot, retain a full packet of credit for
  later rather than burning it.
- Add diagnostic counters for decoder count, prefill candidates, admitted
  packet size, throttled steps, and credit balance. Keep per-step logging off
  by default.

### CPU tests

Extend the scheduler tests to cover:

- The class inherits `AsyncScheduler`.
- C1/C2, C3/C4, and C5 credit rates.
- Immediate admission after an idle or pure-decode period.
- Concurrency transitions without duplicate or lost credit.
- Pure-prefill threshold removal and restoration.
- Approximate global sharing across one to five prefills.
- Six occupied decoder slots do not consume prefill credit.
- Capacity-bound queues cannot bypass a protected decode step.
- Incoming upstream throttles remain authoritative.
- Exceptions from the parent scheduler restore every temporary setting.

## Stage 2: optional exact global-budget hook

Only proceed if Stage 1 improves mixed serving but its fair-share approximation
causes measurable lost throughput or unfairness.

Add a narrow, guarded patch to vLLM's scheduling loop that introduces an
explicit aggregate `mixed_prefill_token_budget`. Both running and newly
admitted prefills decrement the same budget after they are scheduled.

Implement water-filling:

1. Compute a fair share from the remaining global budget and remaining prefill
   candidates.
2. Cap the current prefill at that share.
3. Subtract the tokens actually scheduled.
4. Recompute the fair share for later prefills, automatically redistributing
   unused capacity.

The patch should use guarded source anchors and fail the image build if the
underlying vLLM scheduler changes. It must not copy and permanently fork the
entire scheduler unless a small hook proves impossible. Port only the global
budget/water-fill concept from the DeepSeek recipe; do not copy its cadence of
16 because scheduler-step duration and model throughput differ materially.

## Benchmark plan

Test only the 1M server profile. Use the current Marlin server and identical
model/runtime settings for every scheduler comparison.

### Policies

1. Static baseline: 512 tokens every 4 steps.
2. Adaptive Stage 1.
3. Static 256 tokens every 2 steps as a same-average-work, smoother-latency
   control.
4. Static 512 tokens every 2 steps as a higher-prefill-throughput control.
5. Exact global-budget Stage 2 only if implemented.

### Workloads

For each policy, measure deterministic fresh-prefix workloads at total session
concurrency 2, 3, 4, and 6:

- Long-running decoders only, to establish the no-prefill ceiling.
- One fresh 32K prefill overlapping the remaining decoders.
- One fresh 128K prefill overlapping the remaining decoders.
- Two or more simultaneous prefills where concurrency permits.
- A cached-prefix follow-up to ensure scheduler changes do not defeat prefix
  reuse.

Warm kernels before collecting results. Use distinct prompt prefixes between
runs to prevent accidental cache hits, and keep generation parameters and
output lengths identical.

### Metrics

Record both aggregate and per-request results:

- Decode tokens/s per stream and aggregate.
- DFlash acceptance rate and accepted tokens per engine step.
- Median, p95, and maximum inter-token latency.
- Longest visible decode pause during an admitted prefill chunk.
- Prefill TTFT and input tokens/s.
- Request completion latency.
- Aggregate useful tokens/s.
- Scheduler packet admissions and actual mixed-prefill tokens scheduled.
- Prefix-cache hit tokens.
- Errors, preemptions, OOMs, and server health.

Iteration cadence alone is not a latency metric: a mixed iteration may take
far longer than a decode-only iteration.

## Acceptance gates

Promote an adaptive policy only if all of the following hold:

- No correctness errors, DFlash lifecycle regressions, or OOMs.
- KV capacity is unchanged apart from negligible scheduler bookkeeping.
- Pure-prefill throughput is within 3% of the current server.
- C1/C2 decode-only performance is within 3% of the static baseline.
- Under mixed workloads, p95 inter-token latency remains interactive and the
  maximum decode pause does not materially exceed the agreed user-facing SLO.
- At concurrency 3-6, either prefill TTFT or aggregate useful throughput
  improves materially; target at least 15% to justify added complexity.
- DFlash acceptance does not regress beyond ordinary run-to-run variance.

Before the A/B, choose an explicit interactive latency target. A reasonable
starting gate is p95 inter-token latency below 750 ms and no recurring
multi-second stalls, but observed baseline distributions should determine the
final threshold.

## Rollout and rollback

1. Implement and unit-test Stage 1 without touching the live server.
2. Commit the opt-in implementation while `static` remains the launch default.
3. During an approved test window, collect the static baseline first.
4. Restart with only the scheduler policy changed and run the matched matrix.
5. Restore the static server immediately if health, output quality, acceptance,
   or decode latency fails a gate.
6. Promote adaptive mode to the recipe default only after repeatable wins at
   multiple concurrency levels.
7. Keep `PREFILL_ADMISSION_POLICY=static` as a permanent one-variable rollback.

Do not change the MoE backend, quantization, KV format, DFlash K, memory
utilization, sequence limit, or network configuration during this scheduler
A/B. Those changes would make the result uninterpretable.

## Expected result

The plug-in-only stage is likely sufficient to improve utilization and TTFT at
three to six mixed sessions because it admits more prefill work precisely when
decode batching is already healthier. Its global-share approximation should
also reduce latency spikes when several prefills coexist. The exact vLLM hook
is an optimization for fairness and reclaiming unused shares, not a prerequisite
for determining whether adaptive admission is valuable on this cluster.
