# NVFP4 weight optimization

This recipe improves the routed-expert NVFP4 group scales, packed codes, and
compatible FP32 global divisors without changing the compressed-tensors layout, tensor
shapes or dtypes, Marlin W4A16 kernels, KV cache, DFlash2 configuration, or
tensor payload size. It is an offline checkpoint transformation and adds no
serving work.

## Method and measured result

The tooling also supports direct requantization from the official BF16
checkpoint without changing the existing block-FP8 default. See
[BF16-NVFP4-REQUANTIZATION.md](BF16-NVFP4-REQUANTIZATION.md).

The Red Hat checkpoint derives one FP8 E4M3 scale for every 16 FP4 E2M1
weights using memoryless min/max quantization. For each group, this optimizer
searches the 25 nearby legal E4M3 scale codes (`-16` through `+8`), requantizes
the 16 source values to their nearest E2M1 codes, and selects the scale/code
combination with minimum squared weight error. The source of truth is the
official block-FP8 checkpoint at revision
`03eb5366286afd40d2221b1d9c63a6dd1ba4832e`.

The group-scale phase is a multi-candidate generalization of [Four Over
Six](https://arxiv.org/abs/2512.02010), which compares effective block ranges
`M=6` and `M=4` by reconstruction MSE. A second phase searches 16
log-uniform phases across one octave of each matrix's existing FP32 divisor.
Powers of two are effectively redundant with E4M3, so one octave spans the
useful choices without changing representation.

The divisor phase scores alternating rows from a deterministic 32-row sample
as train and held-out sets. It falls back to the phase-one tensor on any
held-out regression and repeats the gate over the complete matrix before
writing. This makes the optimization non-regressive under its weight-MSE
objective while retaining exactly the existing serving layout.

On 72 matrices spanning layers 3, 23, and 43, eight widely separated experts,
and gate/up/down projections, every phase-one matrix improved:

| metric | Red Hat | group-scale | independent divisor | Red Hat → independent |
|---|---:|---:|---:|---:|
| Mean weight relative RMSE | 9.1114% | 7.7185% | 7.6630% | 15.8968% |
| Mean random-input output RMSE | 9.1182% | 7.7211% | 7.6633% | 15.9559% |

The divisor phase alone reduced representative weight RMSE another 0.7187%
and weight MSE 1.4323%; random-input output RMSE fell another 0.7482%.

Those independent-divisor numbers are an offline upper-bound experiment, not
the promoted Marlin representation. The compressed-tensors fused-MoE loader
combines gate and up as W13 and carries one global divisor for the pair. An
independently selected gate/up pair can therefore be reconstructed with the
wrong scale at runtime. The promoted checkpoint restores gate/up from the
phase-one artifact as complete matched triplets, retaining their 7.72% result,
and keeps the second divisor phase only for unfused down projections. This has
the same size and serving performance while satisfying Marlin exactly.

On the same 72-matrix sample (37.75 million group scales), 65.02% of stored
scales change. The median across all scales is unchanged and the mean is
1.180x Red Hat. Among changed scales, the median is 1.385x and the mean is
1.277x; the 10th/90th percentiles are 0.909x/1.556x. Packed FP4 codes are
requantized with each scale, so this is not a global multiplication of the
weights.

The representative report is
`accuracy-campaign/results/rounding-representative-8experts/report.json`;
the checked-in divisor summary is
`accuracy-campaign/results/global-divisor-representative-v1-summary.json`.

## Full-build validation

The phase-one build completed all 36,288 routed-expert matrices (72,576
replacement tensors). Its structural verifier passed all ten shards and
exactly matched all 144 tensors in the independent 72-matrix reference. The
phase-one serving suite passed 16/16 checks, including 41K-token retrieval and
cached replay.

The final global-divisor build was produced independently across the two
nodes, exchanged, and SHA-256 verified: every one of the ten shard hashes
matched. The semantic audit passed with all 36,288 matrices and 108,864 target
tensors covered. It also checked the unchanged index/config, every tensor
name/shape/dtype, equal tensor-payload bytes per shard, sampled changed target
tensors, and sampled unchanged tensors.

Across 304,405,807,104 expert-weight values, weighted MSE changed from
`2.1460127204e-6` to `2.1087078528e-6`: a 1.7383% MSE reduction and
0.8730% RMSE reduction over phase one. New divisors were selected for 36,271
matrices. Eight held-out gates and four complete-matrix gates fell back; no
matrix was allowed to regress against the phase-one tensor.

Machine-readable results are
`accuracy-campaign/results/optimized-full-verification.json` for phase one,
`accuracy-campaign/results/global-divisor-representative-v1-summary.json`,
and `accuracy-campaign/results/global-divisor-full-v1-summary.json`.

After validating the intermediate global-divisor checkpoint, build the
Marlin-safe serving checkpoint on both nodes:

```bash
CURRENT_HOST_PATH=/path/to/glm53-nvfp4-global-divisor-v1 \
REFERENCE_HOST_PATH=/path/to/glm53-redhat-nvfp4-fp8-passthrough-v1 \
OUTPUT_HOST_PATH=/path/to/glm53-nvfp4-marlin-shared-w13-v1 \
IMAGE=glm53-v14:nvfp4-gscale-tooling \
  ./repair-w13-shared-scale-checkpoint.sh
```

The repair's validator checks equality of all 72,576 restored gate/up tensors
and exact equality of all 12,096 paired global scales.

## Build selected checkpoint shards

The official source and phase-one target must both be local. The output is a
sparse directory initially: the wrapper copies only configuration/tokenizer
files and writes complete optimized model shards directly.

```bash
SOURCE_HOST_PATH=/path/to/zai-org-GLM-5.3-Flash-03eb536 \
TARGET_HOST_PATH=/path/to/glm53-redhat-nvfp4-fp8-passthrough-v1 \
OUTPUT_HOST_PATH=/path/to/glm53-nvfp4-global-divisor-v1 \
TARGET_SHARDS=model-00001-of-00010.safetensors,model-00002-of-00010.safetensors \
GLOBAL_DIVISOR_STEPS_PER_OCTAVE=16 GLOBAL_DIVISOR_SEARCH_ROWS=32 \
GLOBAL_DIVISOR_HELDOUT_TOLERANCE=0 IMAGE=glm53-v14:nvfp4-gscale-tooling \
  ./build-optimized-nvfp4-shards.sh
```

Set `GLOBAL_DIVISOR_STEPS_PER_OCTAVE=0` for the phase-one group-scale build.
The build is resumable at shard granularity. To divide it across two nodes,
build shards 1–5 on one node and 6–10 on the other, then exchange the completed
shard files. Preserve and distinctly name both `nvfp4-build-tensors.jsonl`
reports for coverage and cross-node consistency verification. The complete
ground-up sequence, including the FP8 repair between phases, is in
[CURRENT-RECIPE.md](CURRENT-RECIPE.md).

## Verify

After assembling all ten final shards, run the structural and report verifier:

```bash
docker run --rm --entrypoint /usr/bin/python3 \
  -v "$PWD/probes:/workspace/probes:ro" \
  -v /path/to/glm53-redhat-nvfp4-fp8-passthrough-v1:/base:ro \
  -v /path/to/glm53-nvfp4-global-divisor-v1:/candidate \
  glm53-v14:nvfp4-gscale-tooling \
  /workspace/probes/verify_nvfp4_checkpoint.py \
    --base-root /base --candidate-root /candidate \
    --build-report /candidate/nvfp4-build-tensors-head.jsonl \
    --build-report /candidate/nvfp4-build-tensors-dgx1.jsonl \
    --allow-global-divisor-changes \
    --output /candidate/global-divisor-full-verification.json
```

Also compare SHA-256 for all ten final shard files between nodes.

## Serve

Put the completed checkpoint at the same absolute path on both nodes. The
cluster wrapper forwards `MODEL_HOST_PATH` to both ranks:

```bash
MODEL_HOST_PATH=/path/to/glm53-nvfp4-marlin-shared-w13-v1 \
MAX_MODEL_LEN=1048576 GPU_MEMORY_UTILIZATION=0.87 \
  ./start-cluster.sh
```

All other production defaults remain unchanged, including DCP2, DFlash2 K=7,
calibrated four-over-six native-FP4 MLA KV, and the FP8 indexer. Omit
`MAX_MODEL_LEN` to use the validated 1,048,576-token default.
