# Current reproducible recipe

This branch reproduces the validated 2x DGX Spark profile on `spark-0` and
`dgx1.lan`: optimized routed-expert NVFP4 weights, restored official block-FP8
non-expert weights, FP8 KV/indexer cache, DCP2, DFlash2 K=7, 1M context, and
the async decode-first scheduler.

Model weights are not stored in Git. The scripts deterministically rebuild and
verify them from public checkpoints.

## Pinned inputs

- `zai-org/GLM-5.3-Flash` revision
  `03eb5366286afd40d2221b1d9c63a6dd1ba4832e`; its model index must have SHA-256
  `3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05`.
- `RedHatAI/GLM-5.3-Flash-NVFP4`.
- `incoai/GLM-5.3-Flash-DFlash2` revision
  `bf582e4eacc1810f76656d1811693ff6c6737d2a`.
- Image base digest and SparkInfer commit pinned in
  `build-production-image.sh`.

## Build both serving images

Run on both ARM64 nodes:

```bash
./build-production-image.sh
./build-fp8-passthrough-image.sh
```

This creates `glm53-v11:kvopt-final` and
`glm53-v12:fp8-passthrough`. The cluster wrapper compares the
runtime-defining files across nodes before launch.

## Build the checkpoint

First build all ten optimized NVFP4 shards using
[NVFP4-WEIGHT-OPTIMIZATION.md](NVFP4-WEIGHT-OPTIMIZATION.md). Then:

```bash
SOURCE_HOST_PATH=/path/to/zai-org-GLM-5.3-Flash-03eb536 \
BASE_HOST_PATH="$HOME/.cache/huggingface/glm53-redhat-nvfp4-optimized-scales-v1" \
OUTPUT_HOST_PATH="$HOME/.cache/huggingface/glm53-redhat-nvfp4-fp8-passthrough-v1" \
./build-fp8-passthrough-checkpoint.sh
```

The repair is byte-preserving, not a new quantization pass. It leaves the
optimized routed experts unchanged and restores 179 official block-FP8 weights
plus scales. Its verifier checks the source hash, every tensor byte, and
numeric equivalence. The completed artifact left 146,566 tensors unchanged and
matched all 2,617,245,696 checked BF16 values exactly. Copy the resulting
checkpoint and pinned drafter to the same absolute paths on both nodes.

## Validate the scheduler

The adapter must inherit `AsyncScheduler`. The unit test catches the plain
`Scheduler` regression that disabled DFlash speculation:

```bash
docker run --rm --entrypoint python3 \
  -v "$PWD/docker/glm_decode_first_scheduler.py:/workspace/glm_decode_first_scheduler.py:ro" \
  -v "$PWD/probes/test_decode_first_scheduler.py:/workspace/test_decode_first_scheduler.py:ro" \
  -w /workspace glm53-v12:fp8-passthrough \
  test_decode_first_scheduler.py
```

## Start

On `spark-0`:

```bash
./start-cluster.sh
```

That is equivalent to:

```bash
IMAGE=glm53-v12:fp8-passthrough \
MODEL_HOST_PATH="$HOME/.cache/huggingface/glm53-redhat-nvfp4-fp8-passthrough-v1" \
DRAFT_HOST_PATH="$HOME/.cache/huggingface/glm53-dflash2-bf582e4eacc1810f76656d1811693ff6c6737d2a" \
MAX_MODEL_LEN=1048576 DCP_SIZE=2 USE_FP4_INDEXER_CACHE=0 \
ENABLE_DECODE_FIRST_SCHEDULER=1 PREFILL_SCHEDULE_INTERVAL=4 \
LONG_PREFILL_TOKEN_THRESHOLD=512 ./start-cluster.sh
```

The wrapper starts the worker first, waits for `/health`, and prints the actual
boot KV capacity. API base: `http://10.100.32.1:8000/v1`.

Latest validated boot: 3,079,151 logical KV tokens, 34.1 tok/s fixed C1 decode,
and 45.4% DFlash acceptance. Results vary with prompts, cache state, and UMA
state.

Rollback only the scheduler:

```bash
ENABLE_DECODE_FIRST_SCHEDULER=0 ./start-cluster.sh
```

Rollback to the denser native-FP4 KV profile:

```bash
IMAGE=glm53-v11:kvopt-final \
MODEL_HOST_PATH="$HOME/.cache/huggingface/glm53-redhat-nvfp4-optimized-scales-v1" \
USE_FP4_INDEXER_CACHE=1 ENABLE_DECODE_FIRST_SCHEDULER=0 ./start-cluster.sh
```
