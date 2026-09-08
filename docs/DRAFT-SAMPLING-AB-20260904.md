# Greedy versus probabilistic DFlash2 proposals

Known-good recipe checkpoint `0d9c334` was pushed to `main` and
`feature/exl3-ab-current` before this experiment. The experiment does not change
the target checkpoint, target sampling parameters, precision, or scheduler.

The previous live process omitted `draft_sample_method`, selecting vLLM's
`greedy` default. Candidate B explicitly sets `probabilistic`, retaining
`rejection_sample_method=standard`. Block verification is **not** part of this
experiment. DFlash2 already caches FP32 selector scores along the sampled path;
these are the proposal distribution, not the raw draft LM-head distribution.
Six request slots × seven positions × 154,880 vocabulary entries × four bytes
adds 24.82 MiB/rank of persistent draft-logit storage (other temporaries and
boot profiling can also affect memory).

## Matched protocol

- Same EXL3 target / FP8 KV / MXFP8 drafter, TP2, DCP2, K7, GMU0.87.
- Same fast decode pipeline and M64 no-repack prefill path.
- Target graphs C1, draft graphs C1–C6, unchanged adaptive scheduler.
- Prose/code prompts, C1/C3/C6, 256 generated tokens per request, two rounds.
- Temperature 0.7, top-p 0.95, top-k disabled, neutral repetition penalties.
- Explicit per-request seeds, identical between A/B, different across rounds
  and concurrent streams. Seed equality does not guarantee identical text
  between proposal algorithms.
- Each prompt/concurrency shape gets a 64-token warmup, excluded from results.
- Thinking disabled for this short screen; no claim about all reasoning traffic.
- Reject waves with unexpected completed requests, incomplete generations, no
  speculative verification, or preemptions. Keep throughput and acceptance
  separately; neither changed text nor increased acceptance proves quality.

## Commands

The restart helper stops **both** ranks. It expects the already-built
`glm53-exl3:prefill-norepack-v1` image; its build chain is documented in
`SERVING-CAMPAIGN-20260904.md`. It restores greedy mode if candidate startup
fails. It does not automatically select a performance winner.

```bash
# A was measured on the existing live server, without restarting it.
python3 probes/bench_draft_sampling.py --label greedy-A \
  --output results/serving-20260904/draft-sampling-A-t07.json

bash probes/restart_draft_sampling.sh probabilistic
python3 probes/validate_kv_candidate.py
python3 probes/bench_draft_sampling.py --label probabilistic-B \
  --output results/serving-20260904/draft-sampling-B-t07.json

python3 probes/bench_draft_sampling.py --compare \
  results/serving-20260904/draft-sampling-A-t07.json \
  results/serving-20260904/draft-sampling-B-t07.json

# Explicit rollback if needed:
bash probes/restart_draft_sampling.sh greedy
```

## Control results

A completed all twelve measured waves without preemptions or unexpected
traffic. Mean aggregate output rates across the two rounds:

| Concurrency | Prose tok/s | Code tok/s |
| --- | ---: | ---: |
| 1 | 28.71 | 36.05 |
| 3 | 47.66 | 52.12 |
| 6 | 75.44 | 69.71 |

## Candidate results at temperature 0.7

B passed short functional tests, 27,049-token retrieval and an observed
16,384-token prefix hit. All twelve measured waves completed without
preemptions or unexpected traffic. Mean aggregate rates:

| Concurrency | Prose A → B tok/s | Code A → B tok/s |
| --- | ---: | ---: |
| 1 | 28.71 → 25.96 (-9.57%) | 36.05 → 32.68 (-9.34%) |
| 3 | 47.66 → 51.75 (+8.59%) | 52.12 → 53.18 (+2.04%) |
| 6 | 75.44 → 76.66 (+1.63%) | 69.71 → 73.93 (+6.05%) |

C1 prose acceptance fell from 36.71% to 30.08%; code from 48.38% to
42.32%. Wall time per draft cycle (including first-token work) was not worse:
prose 123.99 → 119.62 ms, code 121.45 → 120.91 ms. The C1 regression is
therefore primarily an accepted-length result, not evidence of excessive
sampling-kernel overhead. Different generated continuations still confound
precise attribution; two short seeds are not a general workload benchmark.

Probabilistic proposals are not guaranteed to beat greedy proposals: a diffuse
or mismatched proposal distribution may accept less often than a good modal
guess. The ideal matched-distribution example is not a performance guarantee
for these learned selector scores. No numerical/model-quality conclusion
follows from the acceptance change.

B advertised 4,543,829 logical KV tokens versus 4,879,088 on the original
boot (-6.87%). This is larger than the persistent probability buffer alone
explains. Its limiting worker reported 16,106,255,627 available KV bytes,
4,320,944,128 bytes non-Torch increase, 4,443,510,784 bytes Torch peak increase,
and 896,258,048 bytes graph reservation. Do not label the entire observed KV
difference an inherent cost of probabilistic sampling; a restored-control
memory profile is needed. Filtered accounting records are saved for both ranks.

## Default-temperature follow-up and reverse control

The checkpoint's generation config specifies temperature 1.0 / top-p 0.95.
We therefore also ran two C1 rounds at temperature 1.0 on B, followed by a
greedy restore and the exact matching requests. All functional restore checks
and measured request/preemption gates passed.

| C1 workload, temperature 1.0 | Greedy tok/s | Probabilistic tok/s | Change |
| --- | ---: | ---: | ---: |
| Prose | 28.30 | 31.10 | +9.88% |
| Code | 28.14 | 30.46 | +8.22% |

This is a promising **temperature-dependent** result, not a universal win or
a large-sample confidence estimate. The C1 code samples varied widely with
seed/continuation. No C3/C6 temperature-1.0 comparison was run.

The restored greedy server also repeated both temperature-0.7 C1 rounds:
prose averaged 29.88 tok/s, versus B's 25.96 (-13.10%); code averaged 32.21,
versus B's 32.68 (+1.48%). Thus the prose regression survived the reverse
control, but the original code regression did **not**. Fixed seeds did not
ensure identical completions across boots, and the first code result must
not be presented as a stable penalty. The derived B C1-only JSON explicitly
identifies its source; the original full report is unchanged.

The restored greedy boot advertises 4,729,291 KV tokens. Its post-profile Torch
allocation is 88,121,446,912 bytes, versus B's 88,147,466,752: exactly the
26,019,840-byte probability buffer difference. Both report the same Torch peak
increase. Non-Torch and graph reservations differ as well, so these boots do
not isolate an inherent KV-capacity penalty beyond the known buffer cost.

Final state: **greedy retained**, with the validated EXL3/FP8/MXFP8 settings,
K7, both prefill improvements, GMU0.87 and HTTP 200. The last check had zero
running/waiting requests and zero preemptions. Probabilistic mode remains
explicitly opt-in through the helper; no serving default is changed. For
temperature-1.0-only traffic it merits a broader follow-up, but this short
screen does not justify promoting it for all chats/agents.

Additional reports: `draft-sampling-t10-comparison.json`,
`draft-sampling-t07-reverse-c1-comparison.json`, both process-config JSONs,
and restored-control memory-accounting logs. All paths are under
`results/serving-20260904`.

## Holdout confirmation protocol

At the user's request, a separate confirmation pass uses six fresh seeds
(19001, 19101, 19201, 19301, 19401, 19501) per prompt, 512 output tokens,
C1, temperature 1.0 and the same other sampling settings. Greedy is measured
first on the restored live server, then probabilistic on one new boot. This
seed set and run count are fixed before seeing the candidate results.

Primary throughput is total generated tokens divided by total wall time,
not the arithmetic mean of per-request rates. `confirm_draft_sampling.py`
reports paired-seed percentile-bootstrap intervals (10,000 resamples, fixed
analysis seed). The preselected confirmation criterion requires the 95%
interval's lower bound to exceed zero **for both prompts**. These intervals
do not estimate cross-boot effects or generalization to other prompts,
reasoning settings, context lengths, or concurrency. No early stopping upon
seeing favorable samples.

```bash
python3 probes/bench_draft_sampling.py --label greedy-confirm-A \
  --temperature 1.0 --levels 1 --rounds 6 --tokens 512 --seed 19001 \
  --output results/serving-20260904/draft-confirm-A-t10.json
bash probes/restart_draft_sampling.sh probabilistic
python3 probes/bench_draft_sampling.py --label probabilistic-confirm-B \
  --temperature 1.0 --levels 1 --rounds 6 --tokens 512 --seed 19001 \
  --output results/serving-20260904/draft-confirm-B-t10.json
python3 probes/confirm_draft_sampling.py \
  results/serving-20260904/draft-confirm-A-t10.json \
  results/serving-20260904/draft-confirm-B-t10.json
```

### Holdout results

All 24 measured requests (12 per mode) completed at exactly 512 output tokens,
without preemptions or unexpected completed requests. Candidate functional
checks passed. The fixed six-seed analysis produced:

| C1 prompt | Greedy tok/s | Probabilistic tok/s | Gain | Paired-seed 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Prose | 27.14 | 29.14 | +7.36% | +1.93% to +13.35% |
| Code | 30.21 | 31.93 | +5.72% | -9.54% to +17.17% |

Probabilistic was faster in five of six paired seeds for **each** prompt. The
code exception was substantial (35.43 → 26.49 tok/s), and is retained rather
than discarded as an outlier. Thus prose passed the preselected criterion;
code did not, and `confirmed_both_prompts` is **false**. It would be incorrect
to call the initial 8–10% gain statistically confirmed for both workloads.
The larger holdout supports a more modest prose gain and a positive but
inconclusive code estimate, conditional on these prompts and boots.

Acceptance rose from 31.12% to 34.93% for prose and 37.39% to 40.51% for code.
Wall time per draft cycle was slightly higher, not lower: 117.17 → 118.19 ms
prose and 119.37 → 119.81 ms code. This supports accepted length, rather than
faster kernels, as the mechanism of the observed throughput gain.

Final state after confirmation: **probabilistic proposals running**, standard
rejection sampling, temperature defaults unchanged, K7/EXL3/FP8 KV/MXFP8,
GMU0.87, and 4,615,161 logical KV tokens on this boot. This supersedes the
greedy final state of the earlier screening stage above. Recipe defaults
remain unchanged; the opt-in restart helper reproduces this mode. Neither
the low-temperature regression nor the seed uncertainty has been removed.

Evidence: `draft-confirm-{A,B}-t10.json`, `draft-confirm-comparison.json`,
candidate process/image JSONs and functional/memory logs under
`results/serving-20260904`.
