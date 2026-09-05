# Decode expert-weight reuse audit

2026-09-04, running `glm53-exl3:prefill-norepack-v1`, fast decode enabled,
streaming hint disabled, scratch rows 128. No serving code, weights, KV
configuration, or server lifecycle changed during this audit.

## Result

The proposed within-verification weight-reuse optimization has no remaining
logical packed-weight copy savings for ordinary K=7 C1/C2 decode. This corrects
the earlier hypothesis that overlapping routes might cause repeated expert
weight reads across blocks for those cases. It does **not** establish that
decode as a whole is optimal or measure actual DRAM transactions.

The installed kernel already sorts tokens by expert, assigns each active expert
to one persistent block group, and shares its weights across an M16 row tile.
K=7 verification has at most 8 rows per request: C1 and C2 fit one tile even
when every token selects the same expert. This bound excludes mixed prefill and
different speculative configurations.

Within that tile, the eight blocks partition the flattened K/N tile space into
disjoint intervals. The K32 kernel's second 256-thread subgroup does **not**
duplicate the async copies: `if (sub_k)` returns before the copy instructions.
CPU enumeration of each 16-byte packed-weight copy covers every address exactly
once for both local projection shapes, K4096/N1024 and K1024/N4096. Each
projection is 2 MiB; gate, up and down are distinct weights, totaling 6 MiB per
active expert, per rank, per layer, per M16 pass. Shared gate/up input scales do
not make their weight matrices interchangeable.

## Remaining C3–C6 possibility

When an expert receives more than 16 rows in one invocation, the kernel repeats
the packed-weight pass for the next row tile. A wider M32/M64 kernel could reuse
weights there, at the cost of more activation/accumulator storage and potentially
lower occupancy. It cannot eliminate distinct expert weights or omit rejected
positions whose verification was necessary.

Re-accounting the existing fixed-seed synthetic C6 cases gives:

| Routing pattern | Active experts | M16 passes | Ideal copy-byte reduction |
| --- | ---: | ---: | ---: |
| Uniform | 214 | 214 | 0% |
| Correlated within each 8-token session block | 104 | 104 | 0% |
| Deliberately concentrated on common hot experts | 35 | 47 | 25.53% |

These are **not live model routing distributions**. The last column is the
optimistic removal of repeated logical copies, not a DRAM-bandwidth measurement
or throughput prediction. L2 may already satisfy some repeated reads. No new
kernel or speed win is claimed. A wider-tile prototype should be gated on actual
C3–C6 counts showing enough experts above 16 rows to repay its overhead; it is
not justified as a C1 optimization.

Cross-step weight residency is a separate, unmeasured question. This audit does
not model cache eviction across the intervening model layers or justify keeping
dequantized expert copies in memory.

## Reproduction

The new analysis and tests require only Python's standard library, no GPU or
model allocation:

```bash
python3 -m unittest discover -s probes -p test_exl3_weight_reuse.py
python3 probes/analyze_exl3_weight_reuse.py \
  --synthetic-samples results/serving-20260904/exl3-tickets-forward.jsonl \
                      results/serving-20260904/exl3-tickets-reverse.jsonl
docker exec -i vllm_glm53 python3 - \
  --source /usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext \
  < probes/analyze_exl3_weight_reuse.py
```

Evidence: `results/serving-20260904/exl3-weight-reuse-source-head.json` and
`exl3-weight-reuse-synthetic.json`. Source anchors are drift checks, not formal
verification of arbitrary CUDA code. Read-only hashes on `dgx1.lan` (`spark-1`)
matched the audited head files:

- Fast MoE kernel: `21e581ea3af66142a433a8077c3b23220a525b06170f95cb7bc7002256123c71`
- GEMM inner helper: `c4f938dd53faac6e9732ab7a188531846ce293b690aa3c03d89dddf72d778615`

The live server remained healthy (HTTP 200), with both earlier prefill
improvements and the validated decode pipeline still enabled.
