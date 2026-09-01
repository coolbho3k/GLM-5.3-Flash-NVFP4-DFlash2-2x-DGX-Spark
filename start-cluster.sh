#!/usr/bin/env bash
# Start both TP2 ranks and wait until the OpenAI-compatible API is healthy.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKER_HOST="${WORKER_HOST:-dgx1.lan}"
REMOTE_DIR="${REMOTE_DIR:-$HOME/.cache/glm53-tp2-deploy}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_glm53}"
IMAGE="${IMAGE:-glm53-v11:kvopt-final}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-$HOME/.cache/huggingface/glm53-redhat-nvfp4-optimized-scales-v1}"
API_PORT="${API_PORT:-8000}"
HEAD_IP="${HEAD_IP:-10.100.32.1}"
WORKER_IP="${WORKER_IP:-10.100.32.2}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
POLL_INTERVAL="${POLL_INTERVAL:-20}"
DCP_SIZE="${DCP_SIZE:-2}"
USE_FP4_INDEXER_CACHE="${USE_FP4_INDEXER_CACHE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.87}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)

cleanup_failed_start() {
  status=$?
  trap - ERR INT TERM
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  ssh "${ssh_opts[@]}" "$WORKER_HOST" \
    "docker rm -f $(printf '%q' "$CONTAINER_NAME") >/dev/null 2>&1 || true" || true
  exit "$status"
}

for command in docker ssh scp curl; do
  command -v "$command" >/dev/null || { echo "Required command not found: $command" >&2; exit 1; }
done

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "Head is missing image $IMAGE" >&2
  exit 1
}
ssh "${ssh_opts[@]}" "$WORKER_HOST" \
  "docker image inspect $(printf '%q' "$IMAGE") >/dev/null 2>&1" || {
  echo "$WORKER_HOST is missing image $IMAGE" >&2
  exit 1
}

# Independent builds can have different image IDs while installing identical
# runtime files. Compare the files that define cache layout, DCP behavior,
# compact KDA replay, accounting, and the active SM121 kernels.
runtime_paths=(
  /usr/local/lib/python3.12/dist-packages/b12x/attention/_shared/mla/kv_cache.py
  /usr/local/lib/python3.12/dist-packages/b12x/attention/_shared/mla/prefill.py
  /usr/local/lib/python3.12/dist-packages/b12x/attention/dsa_indexer/sm121_mxfp4.py
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/mamba_utils.py
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/ops/compact_kda_replay.py
  /usr/local/lib/python3.12/dist-packages/vllm/platforms/interface.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/block_table.py
  /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
)
local_runtime="$(
  docker run --rm --entrypoint sha256sum "$IMAGE" "${runtime_paths[@]}"
)"
printf -v remote_checksum_command '%q ' \
  docker run --rm --entrypoint sha256sum "$IMAGE" "${runtime_paths[@]}"
remote_runtime="$(ssh "${ssh_opts[@]}" "$WORKER_HOST" "$remote_checksum_command")"
[[ "$remote_runtime" == "$local_runtime" ]] || {
  echo "$WORKER_HOST does not contain the same validated runtime files" >&2
  diff <(printf '%s\n' "$local_runtime") <(printf '%s\n' "$remote_runtime") || true
  exit 1
}

ssh "${ssh_opts[@]}" "$WORKER_HOST" "mkdir -p $(printf '%q' "$REMOTE_DIR/docker")"
scp "${ssh_opts[@]}" \
  "$SCRIPT_DIR/launch-glm53-vllm-tp2-dflash2.sh" \
  "$SCRIPT_DIR/chat_template_mm.jinja" \
  "$WORKER_HOST:$REMOTE_DIR/"
scp "${ssh_opts[@]}" "$SCRIPT_DIR/docker/sparse_attn_indexer_kpool_sm121.py" \
  "$WORKER_HOST:$REMOTE_DIR/docker/"

# A stale rank can join the wrong rendezvous. Always remove both before a new boot.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
ssh "${ssh_opts[@]}" "$WORKER_HOST" \
  "docker rm -f $(printf '%q' "$CONTAINER_NAME") >/dev/null 2>&1 || true"

echo "Starting worker rank on $WORKER_HOST..."
trap cleanup_failed_start ERR INT TERM
ssh "${ssh_opts[@]}" "$WORKER_HOST" \
  "cd $(printf '%q' "$REMOTE_DIR") && CONTAINER_NAME=$(printf '%q' "$CONTAINER_NAME") IMAGE=$(printf '%q' "$IMAGE") MODEL_HOST_PATH=$(printf '%q' "$MODEL_HOST_PATH") API_PORT=$(printf '%q' "$API_PORT") HEAD_IP=$(printf '%q' "$HEAD_IP") WORKER_IP=$(printf '%q' "$WORKER_IP") DCP_SIZE=$(printf '%q' "$DCP_SIZE") USE_FP4_INDEXER_CACHE=$(printf '%q' "$USE_FP4_INDEXER_CACHE") GPU_MEMORY_UTILIZATION=$(printf '%q' "$GPU_MEMORY_UTILIZATION") MAX_MODEL_LEN=$(printf '%q' "$MAX_MODEL_LEN") CUDA_LAUNCH_BLOCKING=$(printf '%q' "$CUDA_LAUNCH_BLOCKING") ./launch-glm53-vllm-tp2-dflash2.sh 1"
sleep 25
echo "Starting head rank on $(hostname)..."
CONTAINER_NAME="$CONTAINER_NAME" IMAGE="$IMAGE" MODEL_HOST_PATH="$MODEL_HOST_PATH" API_PORT="$API_PORT" \
  HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" \
  DCP_SIZE="$DCP_SIZE" USE_FP4_INDEXER_CACHE="$USE_FP4_INDEXER_CACHE" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING" \
  "$SCRIPT_DIR/launch-glm53-vllm-tp2-dflash2.sh" 0

deadline=$((SECONDS + READY_TIMEOUT))
until curl -fs --max-time 5 "http://127.0.0.1:$API_PORT/health" >/dev/null; do
  local_state="$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}} {{.State.ExitCode}}' 2>/dev/null || true)"
  remote_state="$(ssh "${ssh_opts[@]}" "$WORKER_HOST" \
    "docker inspect $(printf '%q' "$CONTAINER_NAME") --format '{{.State.Status}} {{.State.ExitCode}}' 2>/dev/null" || true)"
  if [[ "$local_state" != running* || "$remote_state" != running* ]]; then
    echo "A rank exited (head: ${local_state:-missing}; worker: ${remote_state:-missing})." >&2
    docker logs --tail 100 "$CONTAINER_NAME" >&2 || true
    ssh "${ssh_opts[@]}" "$WORKER_HOST" "docker logs --tail 100 $(printf '%q' "$CONTAINER_NAME")" >&2 || true
    false
  fi
  (( SECONDS < deadline )) || { echo "Timed out waiting for the API." >&2; false; }
  sleep "$POLL_INTERVAL"
done

echo "GLM-5.3-Flash is healthy at http://$HEAD_IP:$API_PORT/v1"
curl -fsS "http://127.0.0.1:$API_PORT/v1/models"
echo
docker logs "$CONTAINER_NAME" 2>&1 | grep 'GPU KV cache size:' | tail -n 1 || true
trap - ERR INT TERM
