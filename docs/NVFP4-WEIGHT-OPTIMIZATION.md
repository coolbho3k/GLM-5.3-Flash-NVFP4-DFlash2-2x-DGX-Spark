On the same 72-matrix sample (37.75 million group scales), 65.02% of stored
scales change. The median across all scales is unchanged and the mean is
1.180x Red Hat. Among changed scales, the median is 1.385x and the mean is
1.277x; the 10th/90th percentiles are 0.909x/1.556x. Packed FP4 codes are
requantized with each scale, so this is not a global multiplication of the
weights.

# NVFP4 weight-scale optimization

This recipe improves the routed-expert NVFP4 weights without changing the
compressed-tensors layout, tensor shapes or dtypes, global divisors, Marlin
W4A16 kernels, native-FP4 KV/indexer, DFlash2 configuration, or checkpoint
size. It is an offline checkpoint transformation and adds no serving work.

## Method and measured result

The Red Hat checkpoint derives one FP8 E4M3 scale for every 16 FP4 E2M1
weights using memoryless min/max quantization. For each group, this optimizer
searches the 25 nearby legal E4M3 scale codes (`-16` through `+8`), requantizes
the 16 source values to their nearest E2M1 codes, and selects the scale/code
combination with minimum squared weight error. The source of truth is the
official block-FP8 checkpoint at revision
`03eb5366286afd40d2221b1d9c63a6dd1ba4832e`.

This is a multi-candidate generalization of [Four Over
Six](https://arxiv.org/abs/2512.02010), which compares the two effective
block ranges `M=6` and `M=4` by reconstruction MSE. Unlike that paper's full
recipe, this transformation intentionally preserves Red Hat's tensor-wide
FP32 divisor, making the result an exact drop-in for the existing serving
layout.

On 72 matrices spanning layers 3, 23, and 43, eight widely separated experts,
and gate/up/down projections, every matrix improved:

| metric | Red Hat | optimized | relative reduction |
|---|---:|---:|---:|
| Mean weight relative RMSE | 9.1114% | 7.7185% | 15.2743% |
| Mean random-input output RMSE | 9.1182% | 7.7231% | 15.2847% |

On the same 72-matrix sample (37.75 million group scales), 65.02% of stored
scales change. The median across all scales is unchanged and the mean is
1.180x Red Hat. Among changed scales, the median is 1.385x and the mean is
1.277x; the 10th/90th percentiles are 0.909x/1.556x. Packed FP4 codes are
requantized with each scale, so this is not a global multiplication of the
weights.

The representative report is
`accuracy-campaign/results/rounding-representative-8experts/report.json`.

## Full-build validation

The two-node build completed all 36,288 routed-expert matrices (72,576
replacement tensors). The structural verifier passed all ten shards and the
full build exactly matched all 144 scale/packed tensors in the independent
72-matrix reference artifact.

The resulting 1,048,576-token server reported 3,295,524 logical KV tokens
(3.14 full-length requests). The deterministic serving suite passed 16/16
quality checks, including a 41,000-token retrieval and cached replay. Relative
to the Red Hat serving signature, shared-prefix top-1 agreement was 93.08% and
mean top-20 Jensen-Shannon divergence was 0.01254 nats. Aggregate DFlash token
acceptance was 22.02% versus 24.08% for Red Hat; the unchanged draft therefore
agrees slightly less often with the optimized target. The fixed two-round C1
decode benchmark measured 32.8 tok/s versus the prior 33.4 tok/s 1M baseline.

The machine-readable full-checkpoint result is
`accuracy-campaign/results/optimized-full-verification.json`; the serving and
paired comparison artifacts are `optimized-scales-v1.json` and
`optimized-vs-redhat.json` in the same directory.

## Build selected checkpoint shards

The official source and Red Hat target must both be local. The output is a
sparse directory initially: the wrapper copies only configuration/tokenizer
files and writes complete optimized model shards directly.

```bash
SOURCE_HOST_PATH=/path/to/zai-org-GLM-5.3-Flash-03eb536 \
TARGET_HOST_PATH=/path/to/RedHatAI-GLM-5.3-Flash-NVFP4 \
OUTPUT_HOST_PATH=/path/to/GLM-5.3-Flash-NVFP4-optimized-scales-v1 \
TARGET_SHARDS=model-00001-of-00010.safetensors,model-00002-of-00010.safetensors \
  ./build-optimized-nvfp4-shards.sh
```

The build is resumable at shard granularity. To divide it across two nodes,
build shards 1–5 on one node and 6–10 on the other, then exchange the completed
shard files. Preserve both `nvfp4-build-tensors.jsonl` reports for coverage
verification.

## Verify

After assembling all ten candidate shards, run the structural verifier inside
the serving image. The optional reference file proves that the full build is
byte-identical on the original 72-matrix validation sample.

```bash
docker run --rm --entrypoint /usr/bin/python3 \
  -v "$PWD/probes:/workspace/probes:ro" \
  -v /path/to/RedHatAI-GLM-5.3-Flash-NVFP4:/base:ro \
  -v /path/to/GLM-5.3-Flash-NVFP4-optimized-scales-v1:/candidate:ro \
  -v "$PWD/accuracy-campaign/results:/results:ro" \
  glm53-v11:kvopt-final \
  /workspace/probes/verify_nvfp4_checkpoint.py \
    --base-root /base \
    --candidate-root /candidate \
    --build-report /candidate/nvfp4-build-tensors-local.jsonl \
    --build-report /candidate/nvfp4-build-tensors-dgx1.jsonl \
    --reference-replacements \
      /results/rounding-representative-8experts/optimized.safetensors
```

## Serve

Put the completed checkpoint at the same absolute path on both nodes. The
cluster wrapper forwards `MODEL_HOST_PATH` to both ranks:

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash-NVFP4-optimized-scales-v1 \
MAX_MODEL_LEN=1048576 \
  ./start-cluster.sh
```

All other production defaults remain unchanged, including DCP2, DFlash2 K=7,
and the native-FP4 KV/indexer implementation. Omit `MAX_MODEL_LEN` to use the
launcher's 262,144-token default.
