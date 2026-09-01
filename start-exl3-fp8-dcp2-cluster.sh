#!/usr/bin/env bash
# Start the EXL3 + 528-byte FP8 KV + TP2/DCP2 profile on this host and dgx1.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKER_HOST="${WORKER_HOST:-dgx1.lan}"
REMOTE_DIR="${REMOTE_DIR:-$HOME/.cache/glm53-exl3-fp8-dcp2-deploy}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_glm53}"
IMAGE="${IMAGE:-glm53-exl3:fp8-dcp2}"
API_PORT="${API_PORT:-8000}"
HEAD_IP="${HEAD_IP:-10.100.32.1}"
WORKER_IP="${WORKER_IP:-10.100.32.2}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
POLL_INTERVAL="${POLL_INTERVAL:-20}"
MODEL_REVISION="25a44fdbf16862a46b7cc9921142c6c81350af2f"
DRAFT_REVISION="dc77ff1c99eeb2df044ee3d4f0094eb033fee410"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-$HOME/.cache/huggingface/glm53-exl3-tr3-4bpw-$MODEL_REVISION}"
DRAFT_HOST_PATH="${DRAFT_HOST_PATH:-$HOME/.cache/huggingface/glm53-dflash2-$DRAFT_REVISION}"
WORKER_MODEL_HOST_PATH="${WORKER_MODEL_HOST_PATH:-$MODEL_HOST_PATH}"
WORKER_DRAFT_HOST_PATH="${WORKER_DRAFT_HOST_PATH:-$DRAFT_HOST_PATH}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.87}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_DFLASH="${ENABLE_DFLASH:-1}"
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP:-1}"
COMPACT_SPEC_REPLAY="${COMPACT_SPEC_REPLAY:-1}"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)

[[ "$ENABLE_DFLASH" == "0" || "$ENABLE_DFLASH" == "1" ]] || {
  echo "ENABLE_DFLASH must be 0 or 1" >&2
  exit 2
}

cleanup_failed_start() {
  status=$?
  trap - ERR INT TERM
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  ssh "${ssh_opts[@]}" "$WORKER_HOST" "docker rm -f $(printf '%q' "$CONTAINER_NAME") >/dev/null 2>&1 || true" || true
  exit "$status"
}

for command in curl docker scp ssh; do
  command -v "$command" >/dev/null || {
    echo "Required command not found: $command" >&2
    exit 1
  }
done
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "Head is missing $IMAGE; run ./build-exl3-fp8-dcp2-image.sh" >&2
  exit 1
}
local_image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
remote_image_id="$(
  ssh "${ssh_opts[@]}" "$WORKER_HOST" \
    "docker image inspect $(printf '%q' "$IMAGE") --format '{{.Id}}' 2>/dev/null" \
    || true
)"
if [[ "$remote_image_id" != "$local_image_id" ]]; then
  echo "Worker image is missing or stale; streaming $local_image_id to $WORKER_HOST"
  docker save "$IMAGE" | ssh "${ssh_opts[@]}" "$WORKER_HOST" docker load
fi

runtime_paths=(
  /usr/local/lib/python3.12/dist-packages/b12x/attention/_shared/mla/fp8_zero_rope.py
  /usr/local/lib/python3.12/dist-packages/b12x/attention/_shared/mla/traits.py
  /usr/local/lib/python3.12/dist-packages/b12x/attention/_shared/mla/prefill.py
  /usr/local/lib/python3.12/dist-packages/b12x/attention/_shared/mla/prefill_mg.py
  /usr/local/lib/python3.12/dist-packages/b12x/attention/dsa_indexer/sm121_mxfp4.py
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py
  /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/attention.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_attn.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py
)
local_runtime="$(docker run --rm --entrypoint sha256sum "$IMAGE" "${runtime_paths[@]}")"
printf -v remote_checksum_command '%q ' docker run --rm --entrypoint sha256sum "$IMAGE" "${runtime_paths[@]}"
remote_runtime="$(ssh "${ssh_opts[@]}" "$WORKER_HOST" "$remote_checksum_command")"
[[ "$remote_runtime" == "$local_runtime" ]] || {
  echo "Worker runtime files do not match the head image" >&2
  diff <(printf '%s\n' "$local_runtime") <(printf '%s\n' "$remote_runtime") || true
  exit 1
}

test -f "$MODEL_HOST_PATH/config.json"
test -f "$DRAFT_HOST_PATH/model.safetensors"
ssh "${ssh_opts[@]}" "$WORKER_HOST" \
  "test -f $(printf '%q' "$WORKER_MODEL_HOST_PATH/config.json") && test -f $(printf '%q' "$WORKER_DRAFT_HOST_PATH/model.safetensors")"

ssh "${ssh_opts[@]}" "$WORKER_HOST" "mkdir -p $(printf '%q' "$REMOTE_DIR")"
scp "${ssh_opts[@]}" "$SCRIPT_DIR/launch-glm53-exl3-fp8-dcp2.sh" "$SCRIPT_DIR/chat_template_mm.jinja" "$WORKER_HOST:$REMOTE_DIR/"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
ssh "${ssh_opts[@]}" "$WORKER_HOST" "docker rm -f $(printf '%q' "$CONTAINER_NAME") >/dev/null 2>&1 || true"

trap cleanup_failed_start ERR INT TERM
echo "Starting EXL3 worker rank on $WORKER_HOST"
ssh "${ssh_opts[@]}" "$WORKER_HOST" \
  "cd $(printf '%q' "$REMOTE_DIR") && CONTAINER_NAME=$(printf '%q' "$CONTAINER_NAME") IMAGE=$(printf '%q' "$IMAGE") API_PORT=$(printf '%q' "$API_PORT") HEAD_IP=$(printf '%q' "$HEAD_IP") WORKER_IP=$(printf '%q' "$WORKER_IP") MODEL_HOST_PATH=$(printf '%q' "$WORKER_MODEL_HOST_PATH") DRAFT_HOST_PATH=$(printf '%q' "$WORKER_DRAFT_HOST_PATH") GPU_MEMORY_UTILIZATION=$(printf '%q' "$GPU_MEMORY_UTILIZATION") MAX_MODEL_LEN=$(printf '%q' "$MAX_MODEL_LEN") MAX_NUM_SEQS=$(printf '%q' "$MAX_NUM_SEQS") MAX_NUM_BATCHED_TOKENS=$(printf '%q' "$MAX_NUM_BATCHED_TOKENS") ENFORCE_EAGER=$(printf '%q' "$ENFORCE_EAGER") ENABLE_DFLASH=$(printf '%q' "$ENABLE_DFLASH") DFLASH_TOKENS=$(printf '%q' "$DFLASH_TOKENS") DFLASH_DRAFT_TP=$(printf '%q' "$DFLASH_DRAFT_TP") COMPACT_SPEC_REPLAY=$(printf '%q' "$COMPACT_SPEC_REPLAY") ./launch-glm53-exl3-fp8-dcp2.sh 1"
sleep 25

echo "Starting EXL3 head rank on $(hostname)"
CONTAINER_NAME="$CONTAINER_NAME" IMAGE="$IMAGE" API_PORT="$API_PORT" \
  HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" \
  MODEL_HOST_PATH="$MODEL_HOST_PATH" DRAFT_HOST_PATH="$DRAFT_HOST_PATH" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  ENFORCE_EAGER="$ENFORCE_EAGER" ENABLE_DFLASH="$ENABLE_DFLASH" \
  DFLASH_TOKENS="$DFLASH_TOKENS" DFLASH_DRAFT_TP="$DFLASH_DRAFT_TP" \
  COMPACT_SPEC_REPLAY="$COMPACT_SPEC_REPLAY" \
  "$SCRIPT_DIR/launch-glm53-exl3-fp8-dcp2.sh" 0

deadline=$((SECONDS + READY_TIMEOUT))
until curl -fs --max-time 5 "http://127.0.0.1:$API_PORT/health" >/dev/null; do
  local_state="$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}} {{.State.ExitCode}}' 2>/dev/null || true)"
  remote_state="$(ssh "${ssh_opts[@]}" "$WORKER_HOST" "docker inspect $(printf '%q' "$CONTAINER_NAME") --format '{{.State.Status}} {{.State.ExitCode}}' 2>/dev/null" || true)"
  if [[ "$local_state" != running* || "$remote_state" != running* ]]; then
    echo "A rank exited (head: ${local_state:-missing}; worker: ${remote_state:-missing})" >&2
    docker logs --tail 120 "$CONTAINER_NAME" >&2 || true
    ssh "${ssh_opts[@]}" "$WORKER_HOST" "docker logs --tail 120 $(printf '%q' "$CONTAINER_NAME")" >&2 || true
    false
  fi
  (( SECONDS < deadline )) || {
    echo "Timed out waiting for the API" >&2
    false
  }
  sleep "$POLL_INTERVAL"
done

echo "EXL3 + FP8 KV + DCP2 is healthy at http://$HEAD_IP:$API_PORT/v1"
curl -fsS "http://127.0.0.1:$API_PORT/v1/models"
echo
docker logs "$CONTAINER_NAME" 2>&1 | grep 'GPU KV cache size:' | tail -n 1 || true
trap - ERR INT TERM
