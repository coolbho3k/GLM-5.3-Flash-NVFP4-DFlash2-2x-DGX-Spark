#!/usr/bin/env bash
# Restore pre-global-divisor gate/up tensors so Marlin's fused W13 uses one
# exact shared global scale. This performs no new quantization.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-glm53-v14:nvfp4-gscale-tooling}"
CURRENT_HOST_PATH="${CURRENT_HOST_PATH:-$HOME/.cache/huggingface/glm53-nvfp4-global-divisor-v1}"
REFERENCE_HOST_PATH="${REFERENCE_HOST_PATH:-$HOME/.cache/huggingface/glm53-redhat-nvfp4-fp8-passthrough-v1}"
OUTPUT_HOST_PATH="${OUTPUT_HOST_PATH:-$HOME/.cache/huggingface/glm53-nvfp4-marlin-shared-w13-v1}"
TARGET_SHARDS="${TARGET_SHARDS:-}"
NAME="${CONTAINER_NAME:-glm53_w13_scale_repair}"

test -f "$CURRENT_HOST_PATH/model.safetensors.index.json"
test -f "$REFERENCE_HOST_PATH/model.safetensors.index.json"
test "$CURRENT_HOST_PATH" != "$OUTPUT_HOST_PATH"
test "$REFERENCE_HOST_PATH" != "$OUTPUT_HOST_PATH"

if [[ ! -e "$OUTPUT_HOST_PATH" ]]; then
  # Hard-linking makes the initial 175 GB clone instant and space-free. The
  # repair writes each changed shard to a new file and atomically renames it,
  # so no source inode is ever modified in place. The built shards are owned
  # by root, so create the links from the same root container context.
  checkpoint_parent="$(dirname -- "$CURRENT_HOST_PATH")"
  if [[ "$checkpoint_parent" != "$(dirname -- "$OUTPUT_HOST_PATH")" ]]; then
    echo "Current and output checkpoints must share a parent directory" >&2
    exit 1
  fi
  docker run --rm \
    -v "$checkpoint_parent:/checkpoints" \
    --entrypoint /bin/cp \
    "$IMAGE" -al \
    "/checkpoints/$(basename -- "$CURRENT_HOST_PATH")" \
    "/checkpoints/$(basename -- "$OUTPUT_HOST_PATH")"
elif [[ ! -f "$OUTPUT_HOST_PATH/model.safetensors.index.json" ]]; then
  echo "Refusing incomplete existing output directory: $OUTPUT_HOST_PATH" >&2
  exit 1
fi

args=()
if [[ -n "$TARGET_SHARDS" ]]; then
  args+=(--target-shards "$TARGET_SHARDS")
fi

docker run --rm --ipc host \
  --name "$NAME" \
  -v "$SCRIPT_DIR/probes:/workspace/probes:ro" \
  -v "$CURRENT_HOST_PATH:/current:ro" \
  -v "$REFERENCE_HOST_PATH:/reference:ro" \
  -v "$OUTPUT_HOST_PATH:/output" \
  --entrypoint /usr/bin/python3 \
  "$IMAGE" \
  /workspace/probes/repair_nvfp4_w13_shared_scale.py \
    --current-root /current \
    --reference-root /reference \
    --output-root /output \
    "${args[@]}"
