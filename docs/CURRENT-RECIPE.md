# Current reproducible recipe

This branch reproduces the validated 2x DGX Spark profile on `spark-0` and
`dgx1.lan`: optimized routed-expert NVFP4 weights, restored official block-FP8
non-expert weights, a 288-byte native-NVFP4 MLA cache with upstream-style
four-over-six scale search, FP8 indexer cache, DCP2, DFlash2 K=7, 1M context,
and the async decode-first scheduler.

Model weights are not stored in Git. The scripts deterministically rebuild and
verify them from public checkpoints. The promoted artifact preserves Marlin's
fused-W13 contract: every routed-expert gate/up pair has one exact shared FP32
global scale.

## Pinned inputs

- `zai-org/GLM-5.3-Flash` revision
  `03eb5366286afd40d2221b1d9c63a6dd1ba4832e`; its model index must have SHA-256
  `3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05`.
- `RedHatAI/GLM-5.3-Flash-NVFP4`.
- `incoai/GLM-5.3-Flash-DFlash2` revision
  `bf582e4eacc1810f76656d1811693ff6c6737d2a`.
- Image base digest and SparkInfer commit pinned in
  `build-production-image.sh`.

## Build the serving image chain

Run on both ARM64 nodes:

```bash
./build-production-image.sh
./build-fp8-passthrough-image.sh
./build-four-over-six-image.sh
./build-gscale-tooling-image.sh
```

This creates `glm53-v11:kvopt-final`, `glm53-v12:fp8-passthrough`,
`glm53-v13:nvfp4-four-over-six`, and the default
`glm53-v14:nvfp4-gscale-tooling`. The v13 layer changes the 288-byte writer's
scale selection; v14 adds the calibrated per-layer outer-scale loader and
offline calibration tools without changing the cache ABI. The cluster wrapper
compares runtime-defining files across nodes before launch.

## Build the checkpoint

The exact validated artifact uses two weight-optimization phases. Phase one
optimizes each group-16 E4M3 scale with the original tensor divisor. The repair
then restores 179 official block-FP8 tensors. Phase two searches the global
divisor's E4M3-grid phase and retains the phase-one tensor whenever either
safety gate rejects a candidate.

Set these paths on both nodes:

```bash
SOURCE=/path/to/zai-org-GLM-5.3-Flash-03eb536
REDHAT=/path/to/RedHatAI-GLM-5.3-Flash-NVFP4
SCALE="$HOME/.cache/huggingface/glm53-redhat-nvfp4-optimized-scales-v1"
FIXED="$HOME/.cache/huggingface/glm53-redhat-nvfp4-fp8-passthrough-v1"
GLOBAL="$HOME/.cache/huggingface/glm53-nvfp4-global-divisor-v1"
FINAL="$HOME/.cache/huggingface/glm53-nvfp4-marlin-shared-w13-v1"
```

For each optimization phase, assign shards 1–5 to the head and 6–10 to the
worker through the comma-separated `TARGET_SHARDS` variable:

```bash
# Head node:
TARGET_SHARDS=model-00001-of-00010.safetensors,model-00002-of-00010.safetensors,model-00003-of-00010.safetensors,model-00004-of-00010.safetensors,model-00005-of-00010.safetensors

# dgx1:
TARGET_SHARDS=model-00006-of-00010.safetensors,model-00007-of-00010.safetensors,model-00008-of-00010.safetensors,model-00009-of-00010.safetensors,model-00010-of-00010.safetensors
```
Run the group-scale phase on both nodes:

```bash
SOURCE_HOST_PATH="$SOURCE" TARGET_HOST_PATH="$REDHAT" \
OUTPUT_HOST_PATH="$SCALE" TARGET_SHARDS="$TARGET_SHARDS" \
GLOBAL_DIVISOR_STEPS_PER_OCTAVE=0 IMAGE=glm53-v14:nvfp4-gscale-tooling \
./build-optimized-nvfp4-shards.sh
```

Exchange the five completed shards so `SCALE` is complete on both nodes, then
restore the official FP8 tensors on each:

```bash
SOURCE_HOST_PATH="$SOURCE" BASE_HOST_PATH="$SCALE" \
OUTPUT_HOST_PATH="$FIXED" ./build-fp8-passthrough-checkpoint.sh
```

Now run the global-divisor phase, again splitting the ten shards across the two
nodes:

```bash
SOURCE_HOST_PATH="$SOURCE" TARGET_HOST_PATH="$FIXED" \
OUTPUT_HOST_PATH="$GLOBAL" TARGET_SHARDS="$TARGET_SHARDS" \
GLOBAL_DIVISOR_STEPS_PER_OCTAVE=16 GLOBAL_DIVISOR_SEARCH_ROWS=32 \
GLOBAL_DIVISOR_HELDOUT_TOLERANCE=0 IMAGE=glm53-v14:nvfp4-gscale-tooling \
./build-optimized-nvfp4-shards.sh
```

Before exchanging final shards, preserve each node's generic JSONL report as
`nvfp4-build-tensors-head.jsonl` or
`nvfp4-build-tensors-dgx1.jsonl`. Assemble all ten global-divisor shards and both
reports on both nodes, then verify:

```bash
docker run --rm --entrypoint /usr/bin/python3 \
  -v "$PWD/probes:/workspace/probes:ro" \
  -v "$FIXED:/base:ro" -v "$GLOBAL:/candidate" \
  glm53-v14:nvfp4-gscale-tooling \
  /workspace/probes/verify_nvfp4_checkpoint.py \
    --base-root /base --candidate-root /candidate \
    --build-report /candidate/nvfp4-build-tensors-head.jsonl \
    --build-report /candidate/nvfp4-build-tensors-dgx1.jsonl \
    --allow-global-divisor-changes \
    --output /candidate/global-divisor-full-verification.json
```

The verified build covered all 36,288 matrices. Weighted MSE fell 1.7383% and
weighted RMSE 0.8730%; 8 held-out and 4 full-matrix gates safely fell back.
Both nodes' ten shard hashes matched. See
`accuracy-campaign/results/global-divisor-full-v1-summary.json` and
[NVFP4-WEIGHT-OPTIMIZATION.md](NVFP4-WEIGHT-OPTIMIZATION.md).

The compressed-tensors Marlin loader fuses gate and up into W13 and accepts
only one global divisor for the pair. Do not serve the independently optimized
`GLOBAL` intermediate directly. On both nodes, restore the complete matched
gate/up triplets (packed weights, group scales, and global scales) from the
phase-one `FIXED` checkpoint while retaining the global-divisor-optimized
down projections:

```bash
CURRENT_HOST_PATH="$GLOBAL" REFERENCE_HOST_PATH="$FIXED" \
OUTPUT_HOST_PATH="$FINAL" IMAGE=glm53-v14:nvfp4-gscale-tooling \
./repair-w13-shared-scale-checkpoint.sh
```

The repair performs no quantization and is resumable by shard. Its full
verification checks all 72,576 restored tensors and all 12,096 W1/W3 global
scale pairs. The promoted build must report zero mismatches.

## Validate the scheduler

The adapter must inherit `AsyncScheduler`. The unit test catches the plain
`Scheduler` regression that disabled DFlash speculation:

```bash
docker run --rm --entrypoint python3 \
  -v "$PWD/docker/glm_decode_first_scheduler.py:/workspace/glm_decode_first_scheduler.py:ro" \
  -v "$PWD/probes/test_decode_first_scheduler.py:/workspace/test_decode_first_scheduler.py:ro" \
  -w /workspace glm53-v14:nvfp4-gscale-tooling \
  test_decode_first_scheduler.py
```

## Start

On `spark-0`:

```bash
./start-cluster.sh
```

That is equivalent to:

```bash
IMAGE=glm53-v14:nvfp4-gscale-tooling \
MODEL_HOST_PATH="$HOME/.cache/huggingface/glm53-nvfp4-marlin-shared-w13-v1" \
DRAFT_HOST_PATH="$HOME/.cache/huggingface/glm53-dflash2-bf582e4eacc1810f76656d1811693ff6c6737d2a" \
MAX_MODEL_LEN=1048576 DCP_SIZE=2 USE_FP4_INDEXER_CACHE=0 GPU_MEMORY_UTILIZATION=0.87 \
USE_CALIBRATED_NVFP4_MLA=1 \
ENABLE_DECODE_FIRST_SCHEDULER=1 PREFILL_SCHEDULE_INTERVAL=4 \
LONG_PREFILL_TOKEN_THRESHOLD=512 PREFILL_ADMISSION_POLICY=adaptive \
./start-cluster.sh
```

The wrapper starts the worker first, waits for `/health`, and prints the actual
boot KV capacity. API base: `http://10.100.32.1:8000/v1`.

Latest validated 0.87 boot: 3,212,304 logical KV tokens. A cold 176,041-token
retrieval completed correctly in 123.10 seconds, about 1,430 input tok/s; its
follow-up reused 163,840 prefix tokens. The preceding fixed two-round C1
harness measured 37.2 tok/s with 50.4% DFlash token acceptance and zero
failures. Results vary with prompts, cache state, and UMA state; these serving
numbers establish no regression, not a statistically isolated speedup.

The prepared but unlaunched B12X W4A16 A/B is in [../b12x_experiment](../b12x_experiment/README.md).

The numerical and writer-reader validation is recorded in
[NVFP4-KV-FOUR-OVER-SIX.md](NVFP4-KV-FOUR-OVER-SIX.md).

Rollback only four-over-six:

```bash
IMAGE=glm53-v12:fp8-passthrough ./start-cluster.sh
```

Rollback only the scheduler:

```bash
ENABLE_DECODE_FIRST_SCHEDULER=0 ./start-cluster.sh
```

Use the previous static 512/4 adapter while retaining its `AsyncScheduler`
lifecycle:

```bash
PREFILL_ADMISSION_POLICY=static ./start-cluster.sh
```

Rollback to the denser native-FP4 KV profile:

```bash
IMAGE=glm53-v11:kvopt-final \
MODEL_HOST_PATH="$HOME/.cache/huggingface/glm53-redhat-nvfp4-optimized-scales-v1" \
USE_FP4_INDEXER_CACHE=1 ENABLE_DECODE_FIRST_SCHEDULER=0 ./start-cluster.sh
```
