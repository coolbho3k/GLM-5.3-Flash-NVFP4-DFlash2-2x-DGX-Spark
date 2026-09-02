#!/usr/bin/env bash
# Rebuild selected Red Hat NVFP4 safetensor shards with MSE-optimal group-16
# E4M3 scales and optional FP32 global divisors while preserving the
# compressed-tensors/Marlin serving contract.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-glm53-v11:kvopt-final}"
SOURCE_HOST_PATH="${SOURCE_HOST_PATH:?set SOURCE_HOST_PATH to the pinned official FP8 checkpoint}"
TARGET_HOST_PATH="${TARGET_HOST_PATH:-$HOME/.cache/huggingface/glm53-redhat-nvfp4-9eaeadaf026871a90640e32c0604f6ab0b2d641d}"
OUTPUT_HOST_PATH="${OUTPUT_HOST_PATH:?set OUTPUT_HOST_PATH to the optimized checkpoint directory}"
TARGET_SHARDS="${TARGET_SHARDS:?set TARGET_SHARDS to comma-separated Red Hat shard basenames}"
ROW_CHUNK="${ROW_CHUNK:-256}"
GLOBAL_DIVISOR_STEPS_PER_OCTAVE="${GLOBAL_DIVISOR_STEPS_PER_OCTAVE:-0}"
GLOBAL_DIVISOR_SEARCH_ROWS="${GLOBAL_DIVISOR_SEARCH_ROWS:-32}"
GLOBAL_DIVISOR_HELDOUT_TOLERANCE="${GLOBAL_DIVISOR_HELDOUT_TOLERANCE:-0.0}"
NAME="${CONTAINER_NAME:-glm53_nvfp4_checkpoint_builder}"

test -f "$SOURCE_HOST_PATH/model.safetensors.index.json"
test -f "$TARGET_HOST_PATH/model.safetensors.index.json"
test -s "$SCRIPT_DIR/probes/build_nvfp4_checkpoint.py"
test -s "$SCRIPT_DIR/probes/optimize_nvfp4_rounding.py"
mkdir -p "$OUTPUT_HOST_PATH"

# The builder writes complete optimized model shards directly. Only copy the
# small serving metadata; duplicating the 178 GB baseline shards is unnecessary.
rsync -a \
  --exclude='.cache/' \
  --exclude='model-*.safetensors' \
  "$TARGET_HOST_PATH/" "$OUTPUT_HOST_PATH/"

docker run --gpus all --rm \
  --name "$NAME" --ipc host \
  -v "$SCRIPT_DIR/probes:/workspace/probes:ro" \
  -v "$SOURCE_HOST_PATH:/source:ro" \
  -v "$TARGET_HOST_PATH:/target:ro" \
  -v "$OUTPUT_HOST_PATH:/output" \
  --entrypoint /usr/bin/python3 \
  "$IMAGE" \
  /workspace/probes/build_nvfp4_checkpoint.py \
    --source-root /source \
    --target-root /target \
    --output-root /output \
    --target-shards "$TARGET_SHARDS" \
    --scale-radius-below 16 \
    --scale-radius-above 8 \
    --row-chunk "$ROW_CHUNK" \
    --global-divisor-steps-per-octave "$GLOBAL_DIVISOR_STEPS_PER_OCTAVE" \
    --global-divisor-search-rows "$GLOBAL_DIVISOR_SEARCH_ROWS" \
    --global-divisor-heldout-tolerance "$GLOBAL_DIVISOR_HELDOUT_TOLERANCE"
