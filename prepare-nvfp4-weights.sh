#!/usr/bin/env bash
# Download the published optimized NVFP4 target and DFlash2 draft, then sync both nodes.
set -Eeuo pipefail

MODEL_ID="${MODEL_ID:-coolbho3k/GLM-5.3-Flash-NVFP4-Optimized}"
MODEL_REVISION="${MODEL_REVISION:-b919a80ec2fa8959737d37062fb13e7f6a3d11df}"
# Reuse the established directory; fresh installs still fetch MODEL_REVISION.
MODEL_HOST_PATH="${MODEL_HOST_PATH:-$HOME/.cache/huggingface/glm53-nvfp4-marlin-shared-w13-v1}"
DRAFT_ID="${DRAFT_ID:-incoai/GLM-5.3-Flash-DFlash2}"
DRAFT_REVISION="${DRAFT_REVISION:-bf582e4eacc1810f76656d1811693ff6c6737d2a}"
DRAFT_HOST_PATH="${DRAFT_HOST_PATH:-$HOME/.cache/huggingface/glm53-dflash2-$DRAFT_REVISION}"
EXPECTED_DRAFT_SHA256="${EXPECTED_DRAFT_SHA256-}"
WORKER_HOST="${WORKER_HOST:-dgx1.lan}"
SYNC_WORKER="${SYNC_WORKER:-1}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-10}"
EXPECTED_INDEX_SHA256="${EXPECTED_INDEX_SHA256-54a2a07227b26e9e5930d3546fb64258076d514f8fde15740ba30c5113690502}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-2}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if command -v hf >/dev/null; then
  hf_cmd=(hf)
elif command -v huggingface-cli >/dev/null; then
  hf_cmd=(huggingface-cli)
elif [[ ! -f "$MODEL_HOST_PATH/model.safetensors.index.json" || ! -f "$DRAFT_HOST_PATH/model.safetensors" ]]; then
  echo "Install the Hugging Face CLI first: python3 -m pip install --user 'huggingface_hub[cli]'" >&2
  exit 1
else
  hf_cmd=()
fi

count_shards() {
  find "$1" -maxdepth 1 -name '*.safetensors' -type f 2>/dev/null | wc -l
}

mkdir -p "$MODEL_HOST_PATH" "$DRAFT_HOST_PATH"
if [[ "$(count_shards "$MODEL_HOST_PATH")" -lt "$EXPECTED_SHARDS" ]]; then
  echo "Downloading pinned optimized NVFP4 target (~175 GiB) to $MODEL_HOST_PATH"
  "${hf_cmd[@]}" download "$MODEL_ID" --revision "$MODEL_REVISION" \
    --local-dir "$MODEL_HOST_PATH" --max-workers "$HF_MAX_WORKERS"
fi
test -f "$MODEL_HOST_PATH/config.json"
test -f "$MODEL_HOST_PATH/model.safetensors.index.json"
[[ "$(count_shards "$MODEL_HOST_PATH")" -eq "$EXPECTED_SHARDS" ]] || {
  echo "Incomplete NVFP4 checkpoint: expected $EXPECTED_SHARDS safetensors files" >&2
  exit 1
}
if [[ -n "$EXPECTED_INDEX_SHA256" ]]; then
  actual_index_sha256="$(sha256sum "$MODEL_HOST_PATH/model.safetensors.index.json" | awk '{print $1}')"
  [[ "$actual_index_sha256" == "$EXPECTED_INDEX_SHA256" ]] || {
    echo "NVFP4 index hash does not match pinned checkpoint $MODEL_REVISION" >&2
    exit 1
  }
fi

if [[ ! -f "$DRAFT_HOST_PATH/model.safetensors" ]]; then
  echo "Downloading pinned DFlash2 draft to $DRAFT_HOST_PATH"
  "${hf_cmd[@]}" download "$DRAFT_ID" --revision "$DRAFT_REVISION" \
    --local-dir "$DRAFT_HOST_PATH" --max-workers "$HF_MAX_WORKERS"
fi
test -f "$DRAFT_HOST_PATH/config.json"
test -f "$DRAFT_HOST_PATH/model.safetensors"
if [[ -n "$EXPECTED_DRAFT_SHA256" ]]; then
  actual_draft_sha256="$(sha256sum "$DRAFT_HOST_PATH/model.safetensors" | awk '{print $1}')"
  [[ "$actual_draft_sha256" == "$EXPECTED_DRAFT_SHA256" ]] || {
    echo "Draft checkpoint hash does not match pinned revision $DRAFT_REVISION" >&2
    exit 1
  }
fi

if [[ "$SYNC_WORKER" == "1" ]]; then
  command -v rsync >/dev/null
  ssh -o BatchMode=yes "$WORKER_HOST" \
    "mkdir -p $(printf '%q' "$MODEL_HOST_PATH") $(printf '%q' "$DRAFT_HOST_PATH")"
  rsync -a --exclude '/.cache/' --partial --info=progress2 \
    "$MODEL_HOST_PATH/" "$WORKER_HOST:$MODEL_HOST_PATH/"
  rsync -a --exclude '/.cache/' --partial --info=progress2 \
    "$DRAFT_HOST_PATH/" "$WORKER_HOST:$DRAFT_HOST_PATH/"
fi

printf 'MODEL_HOST_PATH=%q\nDRAFT_HOST_PATH=%q\n' "$MODEL_HOST_PATH" "$DRAFT_HOST_PATH"
