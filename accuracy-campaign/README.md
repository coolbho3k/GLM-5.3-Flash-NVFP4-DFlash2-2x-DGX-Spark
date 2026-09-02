# Accuracy campaign artifacts

This directory contains the deterministic serving corpus and numerical reports
used to evaluate the same-format NVFP4 weight optimization.

## Serving signature

Capture two repeats of all eight cases, including top-20 token logprobs,
quality checks, latency, and before/after DFlash counters:

```bash
python3 probes/capture_accuracy_signature.py \
  --label optimized-nvfp4-scales-v1 \
  --output accuracy-campaign/results/optimized-scales-v1.json
```

Compare it with the checked-in Red Hat baseline:

```bash
python3 probes/compare_accuracy_signatures.py \
  --student accuracy-campaign/results/optimized-scales-v1.json \
  --teacher accuracy-campaign/results/redhat-baseline.json \
  --output accuracy-campaign/results/optimized-vs-redhat.json
```

The baseline contains 16/16 quality passes, 1,614 output tokens, mean 15.51
tokens/s, and aggregate DFlash token acceptance of 24.08% on the stable
native-FP4-KV server.

The optimized full-checkpoint run also passed 16/16. It retained 93.08% top-1
agreement with Red Hat along common output prefixes with 0.01254 nats mean
top-20 Jensen-Shannon divergence. Aggregate DFlash acceptance was 22.02%.
The fixed two-round, 400-token C1 serving benchmark measured 32.8 tok/s versus
the committed 33.4 tok/s 1M baseline.

## Representative weight test

`results/rounding-representative-8experts/report.json` covers layers 3, 23,
and 43; experts 0, 1, 17, 63, 127, 191, 255, and 287; and all gate/up/down
projections. Its 72 matrices improved from 9.1114% to 7.7185% mean relative
weight RMSE. The random-input projection output RMSE improved from 9.1182% to
7.7231%.

## Global-divisor result

The 16-phase, 32-row global-divisor search reduced representative weight RMSE
from 7.7185% to 7.6630% (0.7187% relative) and random-input output RMSE from
7.7211% to 7.6633% (0.7482% relative). The cumulative representative reduction
from raw Red Hat is 15.8968% for weight RMSE and 29.2665% for weight MSE.

The full two-node build covered 36,288/36,288 matrices and
304,405,807,104 values. Weighted MSE fell 1.7383% and RMSE 0.8730% relative
to the group-scale checkpoint. It selected new divisors for 36,271 matrices,
with 8 held-out and 4 full-matrix safety fallbacks. All ten shard SHA-256
hashes matched across nodes, and the structural/semantic verifier passed.
Exact representative and full summaries are
`results/global-divisor-representative-v1-summary.json` and

Generated `optimized.safetensors` files are intentionally gitignored because
the largest representative artifact is 340 MB. Regenerate them with
`probes/optimize_nvfp4_rounding.py`; the JSON reports and pinned source revision
contain the reproducible parameters. See
[`docs/NVFP4-WEIGHT-OPTIMIZATION.md`](../docs/NVFP4-WEIGHT-OPTIMIZATION.md) for
the complete checkpoint recipe and serving instructions.
