#!/usr/bin/env bash
# Restore official block-FP8 tensors without changing optimized routed experts.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_HOST_PATH="${SOURCE_HOST_PATH:-}"
BASE_HOST_PATH="${BASE_HOST_PATH:-$HOME/.cache/huggingface/glm53-redhat-nvfp4-optimized-scales-v1}"
OUTPUT_HOST_PATH="${OUTPUT_HOST_PATH:-$HOME/.cache/huggingface/glm53-redhat-nvfp4-fp8-passthrough-v1}"
VERIFY_IMAGE="${VERIFY_IMAGE:-glm53-v12:fp8-passthrough}"
OVERWRITE="${OVERWRITE:-0}"

: "${SOURCE_HOST_PATH:?set SOURCE_HOST_PATH to pinned zai-org/GLM-5.3-Flash revision 03eb536}"

repair_args=(
  --base-root
  "$BASE_HOST_PATH"
  --source-root
  "$SOURCE_HOST_PATH"
  --output-root
  "$OUTPUT_HOST_PATH"
)
if [[ "$OVERWRITE" == "1" ]]; then
  repair_args+=(--overwrite)
fi

python3 "$SCRIPT_DIR/probes/repair_redhat_fp8_passthrough.py" "${repair_args[@]}"

docker run --rm --entrypoint python3 \
  -v "$SCRIPT_DIR:/workspace:ro" \
  -v "$BASE_HOST_PATH:/base:ro" \
  -v "$SOURCE_HOST_PATH:/source:ro" \
  -v "$OUTPUT_HOST_PATH:/candidate" \
  -w /workspace/probes \
  "$VERIFY_IMAGE" \
  verify_redhat_fp8_passthrough.py \
    --base-root /base \
    --source-root /source \
    --candidate-root /candidate \
    --numeric-equivalence \
    --output /candidate/fp8-passthrough-verification.json

echo "Verified repaired checkpoint at $OUTPUT_HOST_PATH"
