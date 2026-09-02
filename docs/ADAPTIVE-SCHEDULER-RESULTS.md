# Adaptive scheduler results

Status: promoted as the default interactive profile on 2026-09-02.

## Recommendation

Use `PREFILL_ADMISSION_POLICY=adaptive` with
`scheduler_profiles/adaptive.json`:

| Active decoders | Base aggregate packet | Interval | Extra-prefill growth |
|---:|---:|---:|---:|
| 1-2 | 256 tokens | 2 steps | none |
| 3-4 | 256 tokens | 4 steps | +64 per prefill after the first, cap 384 |
| 5-6 | 256 tokens | 6 steps | none |

`global_budget=true` shares each aggregate packet across runnable prefills.
Pure prefill remains uncapped and can use the server's full 8,192-token batch
budget. The JSON profile is checked for changes at most once per second, so
these values can be tuned without restarting the model.

This policy is aimed at a local interactive server with up to six chat/agent
sessions. It favors existing decode streams, but grows the mid-tier aggregate
packet only when several new sessions would otherwise make little progress.

## Matched single-prefill measurements

All cases used the same 1M-context Marlin server, DCP2, DFlash K=7, native-FP4
MLA KV, FP8 indexer cache, `MAX_NUM_SEQS=6`, and fresh long-prefix markers.
First-use JIT-contaminated passes were discarded and repeated warm.

| Active decoders | Policy | Mixed decode/stream | Decode retained | Prefill input tok/s | p95 gap | Max gap |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 256/2 | 8.65 | 58.0% | 557 | 0.353 s | 0.439 s |
| 1 | 512/4 control | 9.78 | 50.4% | 607 | 0.501 s | 0.586 s |
| 3 | 384/3 aggressive | 7.20 | 63.7% | 450 | 0.477 s | 0.533 s |
| 3 | 256/4 protected | 9.01 | 80.6% | 425 | 0.398 s | 0.477 s |
| 3 | 384/4 | 8.05 | 70.1% | 415 | 0.470 s | 0.535 s |
| 5 | 512/2 aggressive | 6.36 | 73.9% | 621 | 0.568 s | 0.614 s |
| 5 | 256/4 midpoint | 7.57 | 78.8% | 351 | 0.438 s | 0.484 s |
| 5 | 256/6 protected | 8.05 | 85.0% | 359 | 0.439 s | 0.497 s |

Absolute speculative-decode rates vary with generated content and acceptance,
even at temperature zero. Within-wave retention and stream-gap measurements
are therefore the primary comparisons. The large protected-versus-aggressive
differences are well outside the observed control variation.

## Six-session saturation

The saturation case ran three 384-token decoders while admitting three fresh
3.4K-5.4K-token prefills. Increasing the aggregate mid-tier packet only when
several prefills compete produced the useful exception to the otherwise
decode-protective policy:

| Aggregate packet | Mixed decode/stream | Decode retained | Aggregate prefill | Max prefill TTFT | p95 gap | Max gap |
|---:|---:|---:|---:|---:|---:|---:|
| 256/4 | 8.62 tok/s | 74.2% | 279 input tok/s | 43.8 s | 0.420 s | 0.751 s |
| 384/4 | 8.97 tok/s | 77.1% | 379 input tok/s | 32.1 s | 0.498 s | 0.992 s |

The 384-token shared packet improved aggregate prefill by 36%, reduced the
slowest prefill TTFT by about 12 seconds, and did not reduce mean decode. It
did make delivery burstier, so the profile enables 384 only when at least
three prefills are competing at the mid concurrency tier.

## Rollback

Use the previous static adapter:

```bash
PREFILL_ADMISSION_POLICY=static ./start-cluster.sh
```

Or disable the adapter entirely:

```bash
ENABLE_DECODE_FIRST_SCHEDULER=0 ./start-cluster.sh
```

Raw machine-readable results are under `results/scheduler-*.json`. The
reusable pressure harness is `probes/bench_scheduler_pressure.py`.
