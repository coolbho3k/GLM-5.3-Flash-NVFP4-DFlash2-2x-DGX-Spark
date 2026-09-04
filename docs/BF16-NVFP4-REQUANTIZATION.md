# BF16-source NVFP4 requantization

Status: tooling is prepared, but no BF16 checkpoint has been downloaded and no
quantization job has been launched.

The source of truth for this experiment is the official
`zai-org/GLM-5.3-Flash-BF16` checkpoint pinned by revision before download.
The output remains the same compressed-tensors NVFP4/Marlin W4A16 layout as the
current recipe. There is no serving-kernel or checkpoint-size change.

## Implemented source modes

`probes/optimize_nvfp4_rounding.py` and
`probes/build_nvfp4_checkpoint.py` accept:

- `--source-format block-fp8` (the existing default), which reconstructs the
  official 128x128 block-FP8 tensors using `weight_scale_inv`.
- `--source-format bf16`, which reads each expert weight directly, requires
  the tensor to be BF16, and deliberately rejects FP8/F32 tensors.

The full-checkpoint wrapper exposes the same choice as
`SOURCE_FORMAT=block-fp8|bf16`. Reports and resumable build state record the
selected source format, and a partially completed output cannot be resumed
under a different format.

## Inventory required BF16 shards without building

After placing only the two checkpoint indexes at their intended source and
target paths, this command prints the exact BF16 shard basenames needed for the
selected output shards. It does not request GPU access, read model shards, or
write an output checkpoint:

```bash
SOURCE_HOST_PATH=/path/to/zai-org-GLM-5.3-Flash-BF16 \
SOURCE_FORMAT=bf16 \
TARGET_HOST_PATH=/path/to/current-optimized-nvfp4-template \
TARGET_SHARDS=model-00001-of-00010.safetensors,model-00002-of-00010.safetensors \
PLAN_ONLY=1 \
  ./build-optimized-nvfp4-shards.sh
```

Run the plan once for target shards 1-5 and once for 6-10 before downloading
the 643 GB source. That permits each node to fetch only its required source
shards.

## Representative pilot

The first real job should be the existing 72-matrix sample against BF16:
layers 3, 23, and 43; experts 0, 1, 17, 63, 127, 191, 255, and 287; and all
three expert projections.

```bash
docker run --gpus all --rm \
  -v "$PWD/probes:/workspace/probes:ro" \
  -v /path/to/zai-org-GLM-5.3-Flash-BF16:/source:ro \
  -v /path/to/current-optimized-nvfp4-template:/target:ro \
  -v /path/to/results:/output \
  --entrypoint /usr/bin/python3 \
  glm53-v14:nvfp4-gscale-tooling \
  /workspace/probes/optimize_nvfp4_rounding.py \
    --source-root /source --source-format bf16 \
    --target-root /target --output-dir /output \
    --layers 3,23,43 --experts 0,1,17,63,127,191,255,287 \
    --projections gate_proj,up_proj,down_proj \
    --global-divisor-steps-per-octave 0
```

Compare the current template and candidate against the same BF16 reference.
Do not compare a BF16-relative RMSE directly with the existing 7.7185% number,
which uses the official FP8 checkpoint as its reference.

## Full two-node build

Use the current optimized checkpoint as the serving-layout template. Split
target shards 1-5 and 6-10 between the two nodes as in the existing recipe.

Phase one creates BF16-derived group scales and packed codes:

```bash
SOURCE_HOST_PATH=/path/to/zai-org-GLM-5.3-Flash-BF16 \
SOURCE_FORMAT=bf16 \
TARGET_HOST_PATH=/path/to/current-optimized-nvfp4-template \
OUTPUT_HOST_PATH=/path/to/glm53-nvfp4-bf16-scale-v1 \
TARGET_SHARDS="$TARGET_SHARDS" \
GLOBAL_DIVISOR_STEPS_PER_OCTAVE=0 \
  ./build-optimized-nvfp4-shards.sh
```

After assembling and verifying all ten phase-one shards, phase two searches
the matrix divisor:

```bash
SOURCE_HOST_PATH=/path/to/zai-org-GLM-5.3-Flash-BF16 \
SOURCE_FORMAT=bf16 \
TARGET_HOST_PATH=/path/to/glm53-nvfp4-bf16-scale-v1 \
OUTPUT_HOST_PATH=/path/to/glm53-nvfp4-bf16-global-v1 \
TARGET_SHARDS="$TARGET_SHARDS" \
GLOBAL_DIVISOR_STEPS_PER_OCTAVE=16 \
GLOBAL_DIVISOR_SEARCH_ROWS=32 \
GLOBAL_DIVISOR_HELDOUT_TOLERANCE=0 \
  ./build-optimized-nvfp4-shards.sh
```

The existing independent divisor search is not safe for fused W13. Until a
joint W1/W3 search is implemented, restore both gate/up triplets from the
BF16-derived phase-one checkpoint:

```bash
CURRENT_HOST_PATH=/path/to/glm53-nvfp4-bf16-global-v1 \
REFERENCE_HOST_PATH=/path/to/glm53-nvfp4-bf16-scale-v1 \
OUTPUT_HOST_PATH=/path/to/glm53-nvfp4-bf16-marlin-safe-v1 \
IMAGE=glm53-v14:nvfp4-gscale-tooling \
  ./repair-w13-shared-scale-checkpoint.sh
```

The regular structural verifier, shard hashes, numerical audits, serving
signature, long-context retrieval, and DFlash acceptance checks still apply.

## Deferred quality extensions

Neither extension below requires a Marlin change:

- Activation-aware group scoring needs a calibration collector and a stable
  on-disk schema for input second moments. The quantizer already accepts an
  `input_second_moment` vector, but the builders intentionally pass `None`
  until those statistics exist.
- Joint W1/W3 divisor search needs to load a gate/up pair, score one shared
  divisor against their combined objective, apply the full-matrix safety gate
  to both, and emit identical global divisors. The resulting tensors already
  satisfy Marlin's fused-W13 contract.
