#!/usr/bin/env bash
# Download the pinned EXL3 target and DFlash2 draft, then sync both to dgx1.
set -Eeuo pipefail

MODEL_ID="${MODEL_ID:-Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw}"
MODEL_REVISION="${MODEL_REVISION:-25a44fdbf16862a46b7cc9921142c6c81350af2f}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-$HOME/.cache/huggingface/glm53-exl3-tr3-4bpw-$MODEL_REVISION}"
DRAFT_ID="${DRAFT_ID:-incoai/GLM-5.3-Flash-DFlash2}"
DRAFT_REVISION="${DRAFT_REVISION:-bf582e4eacc1810f76656d1811693ff6c6737d2a}"
DRAFT_HOST_PATH="${DRAFT_HOST_PATH:-$HOME/.cache/huggingface/glm53-dflash2-$DRAFT_REVISION}"
EXPECTED_DRAFT_SHA256="${EXPECTED_DRAFT_SHA256-}"
WORKER_HOST="${WORKER_HOST:-dgx1.lan}"
SYNC_WORKER="${SYNC_WORKER:-1}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-120}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-2}"
# The live 320B server leaves little host memory. Xet high-performance mode
# consumed ~4.6 GiB RSS here; plain HTTP with two workers stayed near 125 MiB.
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
  echo "Downloading pinned EXL3 target (~164 GiB) to $MODEL_HOST_PATH"
  "${hf_cmd[@]}" download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_HOST_PATH" \
    --max-workers "$HF_MAX_WORKERS" \
    --exclude 'runtime-results/**' \
    --exclude 'src/**' \
    --exclude 'runtime/src/**' \
    --exclude 'scripts/**' \
    --exclude 'docs/**' \
    --exclude 'results/**'
fi
test -f "$MODEL_HOST_PATH/config.json"
test -f "$MODEL_HOST_PATH/model.safetensors.index.json"
[[ "$(count_shards "$MODEL_HOST_PATH")" -ge "$EXPECTED_SHARDS" ]] || {
  echo "Incomplete EXL3 checkpoint: expected at least $EXPECTED_SHARDS safetensors files" >&2
  exit 1
}

if [[ ! -f "$DRAFT_HOST_PATH/model.safetensors" ]]; then
  echo "Downloading pinned DFlash2 draft to $DRAFT_HOST_PATH"
  "${hf_cmd[@]}" download "$DRAFT_ID" --revision "$DRAFT_REVISION" --local-dir "$DRAFT_HOST_PATH" --max-workers "$HF_MAX_WORKERS"
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
  ssh -o BatchMode=yes "$WORKER_HOST" "mkdir -p $(printf '%q' "$MODEL_HOST_PATH") $(printf '%q' "$DRAFT_HOST_PATH")"
  # `hf download --local-dir` leaves transport metadata under `.cache/`.
  # It is not needed for offline serving and can contain root-owned files when
  # an earlier download ran in a container, so never make it part of the
  # reproducible runtime tree.
  rsync -a --exclude '/.cache/' --partial --info=progress2 \
    "$MODEL_HOST_PATH/" "$WORKER_HOST:$MODEL_HOST_PATH/"
  rsync -a --exclude '/.cache/' --partial --info=progress2 \
    "$DRAFT_HOST_PATH/" "$WORKER_HOST:$DRAFT_HOST_PATH/"
fi

printf 'MODEL_HOST_PATH=%q\nDRAFT_HOST_PATH=%q\n' "$MODEL_HOST_PATH" "$DRAFT_HOST_PATH"
