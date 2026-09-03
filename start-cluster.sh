#!/usr/bin/env bash
# Start both TP2 ranks and wait until the OpenAI-compatible API is healthy.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKER_HOST="${WORKER_HOST:-dgx1.lan}"
REMOTE_DIR="${REMOTE_DIR:-$HOME/.cache/glm53-tp2-deploy}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_glm53}"
IMAGE="${IMAGE:-glm53-v14:nvfp4-gscale-tooling}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-$HOME/.cache/huggingface/glm53-nvfp4-marlin-shared-w13-v1}"
DRAFT_HOST_PATH="${DRAFT_HOST_PATH:-$HOME/.cache/huggingface/glm53-dflash2-bf582e4eacc1810f76656d1811693ff6c6737d2a}"
API_PORT="${API_PORT:-8000}"
HEAD_IP="${HEAD_IP:-10.100.32.1}"
WORKER_IP="${WORKER_IP:-10.100.32.2}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
POLL_INTERVAL="${POLL_INTERVAL:-20}"
DCP_SIZE="${DCP_SIZE:-2}"
USE_FP4_INDEXER_CACHE="${USE_FP4_INDEXER_CACHE:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.87}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
BLOCK_SIZE="${BLOCK_SIZE:-2304}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
QUANTIZATION="${QUANTIZATION-}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND-}"
CACHE_HOST_PATH="${CACHE_HOST_PATH:-$HOME/.cache/huggingface/vllm-cache-glm53}"
JIT_CACHE_HOST_PATH="${JIT_CACHE_HOST_PATH:-$HOME/.cache/glm53-vllm-jit}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
COMPILATION_CONFIG="${COMPILATION_CONFIG-}"
ENABLE_DFLASH="${ENABLE_DFLASH:-1}"
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP-}"
DFLASH_DRAFT_SAMPLE_METHOD="${DFLASH_DRAFT_SAMPLE_METHOD-}"
DFLASH_REJECTION_SAMPLE_METHOD="${DFLASH_REJECTION_SAMPLE_METHOD-}"
EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"
EXL3_MOE_ROW_TILE="${EXL3_MOE_ROW_TILE:-0}"
EXL3_FAT_KERNEL="${EXL3_FAT_KERNEL:-1}"
EXL3_TEMP_ROWS_FUSED="${EXL3_TEMP_ROWS_FUSED:-128}"
COMPACT_SPEC_REPLAY="${COMPACT_SPEC_REPLAY:-1}"
GLM53_SPINWAIT_MS="${GLM53_SPINWAIT_MS:-stock}"
VLLM_B12X_USE_CUDA_GRAPH="${VLLM_B12X_USE_CUDA_GRAPH:-0}"
VLLM_B12X_CUDA_GRAPH_MAX_TOKENS="${VLLM_B12X_CUDA_GRAPH_MAX_TOKENS:-64}"
SYNC_IMAGE_TO_WORKER="${SYNC_IMAGE_TO_WORKER:-0}"
ENABLE_DECODE_FIRST_SCHEDULER="${ENABLE_DECODE_FIRST_SCHEDULER:-1}"
PREFILL_ADMISSION_POLICY="${PREFILL_ADMISSION_POLICY:-adaptive}"
PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-4}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-512}"
ADAPTIVE_SCHEDULER_CONFIG="${ADAPTIVE_SCHEDULER_CONFIG:-$SCRIPT_DIR/scheduler_profiles/adaptive.json}"
ADAPTIVE_SCHEDULER_RELOAD_SECONDS="${ADAPTIVE_SCHEDULER_RELOAD_SECONDS:-1}"
USE_CALIBRATED_NVFP4_MLA="${USE_CALIBRATED_NVFP4_MLA:-1}"
if [[ "$USE_CALIBRATED_NVFP4_MLA" == "1" ]]; then
  NVFP4_CALIBRATION_HOST_PATH="${NVFP4_CALIBRATION_HOST_PATH:-$SCRIPT_DIR/kv_calibration/results}"
  NVFP4_MLA_SCALES_FILE="${NVFP4_MLA_SCALES_FILE:-nvfp4-gscale-v1.json}"
elif [[ "$USE_CALIBRATED_NVFP4_MLA" == "0" ]]; then
  NVFP4_CALIBRATION_HOST_PATH="${NVFP4_CALIBRATION_HOST_PATH:-}"
  NVFP4_MLA_SCALES_FILE=""
else
  echo "USE_CALIBRATED_NVFP4_MLA must be 0 or 1" >&2
  exit 2
fi
REMOTE_NVFP4_CALIBRATION_HOST_PATH="${REMOTE_NVFP4_CALIBRATION_HOST_PATH:-$REMOTE_DIR/nvfp4-calibration}"
ENABLE_NVFP4_MLA_CAPTURE="${ENABLE_NVFP4_MLA_CAPTURE:-0}"
NVFP4_MLA_CAPTURE_GROUPS_PER_STRATUM="${NVFP4_MLA_CAPTURE_GROUPS_PER_STRATUM:-65536}"
NVFP4_MLA_CAPTURE_GROUPS_PER_CALL="${NVFP4_MLA_CAPTURE_GROUPS_PER_CALL:-1024}"
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
if ! ssh "${ssh_opts[@]}" "$WORKER_HOST" \
  "docker image inspect $(printf '%q' "$IMAGE") >/dev/null 2>&1"; then
  if [[ "$SYNC_IMAGE_TO_WORKER" == "1" ]]; then
    echo "$WORKER_HOST is missing $IMAGE; streaming it from the head"
    docker save "$IMAGE" | ssh "${ssh_opts[@]}" "$WORKER_HOST" docker load
  else
    echo "$WORKER_HOST is missing image $IMAGE (set SYNC_IMAGE_TO_WORKER=1 to copy it)" >&2
    exit 1
  fi
fi

# Independent builds can have different image IDs while installing identical
# runtime files. Compare the files that define cache layout, DCP behavior,
# compact KDA replay, accounting, and the active SM121 kernels.
runtime_paths=(
  /usr/local/lib/python3.12/dist-packages/b12x/attention/_shared/mla/kv_cache.py
  /usr/local/lib/python3.12/dist-packages/b12x/attention/_shared/mla/prefill.py
  /usr/local/lib/python3.12/dist-packages/b12x/attention/dsa_indexer/sm121_mxfp4.py
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/mamba_utils.py
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
  /usr/local/lib/python3.12/dist-packages/vllm/config/scheduler.py
  /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/ops/compact_kda_replay.py
  /usr/local/lib/python3.12/dist-packages/vllm/platforms/interface.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/block_table.py
  /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py
  /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py
  /usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/b12x_moe.py
  /usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py
)
if docker run --rm --entrypoint test "$IMAGE" \
  -f /usr/local/lib/python3.12/dist-packages/vllm/nvfp4_mla_calibration.py; then
  runtime_paths+=(/usr/local/lib/python3.12/dist-packages/vllm/nvfp4_mla_calibration.py)
fi
if docker run --rm --entrypoint test "$IMAGE" \
  -f /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py; then
  runtime_paths+=(/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py)
fi
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

worker_calibration_host_path="$NVFP4_CALIBRATION_HOST_PATH"
if [[ -n "$NVFP4_MLA_SCALES_FILE" ]]; then
  scale_artifact="$NVFP4_CALIBRATION_HOST_PATH/$NVFP4_MLA_SCALES_FILE"
  test -s "$scale_artifact" || {
    echo "Missing NVFP4 MLA scale artifact: $scale_artifact" >&2
    exit 1
  }
  worker_calibration_host_path="$REMOTE_NVFP4_CALIBRATION_HOST_PATH"
  ssh "${ssh_opts[@]}" "$WORKER_HOST" \
    "mkdir -p $(printf '%q' "$worker_calibration_host_path")"
  scp "${ssh_opts[@]}" "$scale_artifact" \
    "$WORKER_HOST:$worker_calibration_host_path/$NVFP4_MLA_SCALES_FILE"
fi

ssh "${ssh_opts[@]}" "$WORKER_HOST" "mkdir -p $(printf '%q' "$REMOTE_DIR/docker")"
scp "${ssh_opts[@]}" \
  "$SCRIPT_DIR/launch-glm53-vllm-tp2-dflash2.sh" \
  "$SCRIPT_DIR/chat_template_mm.jinja" \
  "$WORKER_HOST:$REMOTE_DIR/"
scp "${ssh_opts[@]}" "$SCRIPT_DIR/docker/sparse_attn_indexer_kpool_sm121.py" \
  "$SCRIPT_DIR/docker/glm_decode_first_scheduler.py" \
  "$WORKER_HOST:$REMOTE_DIR/docker/"

worker_adaptive_scheduler_config="$ADAPTIVE_SCHEDULER_CONFIG"
if [[ "$PREFILL_ADMISSION_POLICY" == "adaptive" ]]; then
  test -s "$ADAPTIVE_SCHEDULER_CONFIG"
  ssh "${ssh_opts[@]}" "$WORKER_HOST" "mkdir -p $(printf '%q' "$REMOTE_DIR/scheduler_profiles")"
  scp "${ssh_opts[@]}" "$ADAPTIVE_SCHEDULER_CONFIG" "$WORKER_HOST:$REMOTE_DIR/scheduler_profiles/adaptive.json"
  worker_adaptive_scheduler_config="$REMOTE_DIR/scheduler_profiles/adaptive.json"
fi

# A stale rank can join the wrong rendezvous. Always remove both before a new boot.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
ssh "${ssh_opts[@]}" "$WORKER_HOST" \
  "docker rm -f $(printf '%q' "$CONTAINER_NAME") >/dev/null 2>&1 || true"

echo "Starting worker rank on $WORKER_HOST..."
trap cleanup_failed_start ERR INT TERM
ssh "${ssh_opts[@]}" "$WORKER_HOST" \
  "cd $(printf '%q' "$REMOTE_DIR") && CONTAINER_NAME=$(printf '%q' "$CONTAINER_NAME") IMAGE=$(printf '%q' "$IMAGE") MODEL_HOST_PATH=$(printf '%q' "$MODEL_HOST_PATH") DRAFT_HOST_PATH=$(printf '%q' "$DRAFT_HOST_PATH") API_PORT=$(printf '%q' "$API_PORT") HEAD_IP=$(printf '%q' "$HEAD_IP") WORKER_IP=$(printf '%q' "$WORKER_IP") DCP_SIZE=$(printf '%q' "$DCP_SIZE") USE_FP4_INDEXER_CACHE=$(printf '%q' "$USE_FP4_INDEXER_CACHE") GPU_MEMORY_UTILIZATION=$(printf '%q' "$GPU_MEMORY_UTILIZATION") MAX_MODEL_LEN=$(printf '%q' "$MAX_MODEL_LEN") BLOCK_SIZE=$(printf '%q' "$BLOCK_SIZE") CUDA_LAUNCH_BLOCKING=$(printf '%q' "$CUDA_LAUNCH_BLOCKING") MAX_NUM_BATCHED_TOKENS=$(printf '%q' "$MAX_NUM_BATCHED_TOKENS") MAX_NUM_SEQS=$(printf '%q' "$MAX_NUM_SEQS") QUANTIZATION=$(printf '%q' "$QUANTIZATION") MOE_BACKEND=$(printf '%q' "$MOE_BACKEND") KV_CACHE_DTYPE=$(printf '%q' "$KV_CACHE_DTYPE") VLLM_ATTENTION_BACKEND=$(printf '%q' "$VLLM_ATTENTION_BACKEND") CACHE_HOST_PATH=$(printf '%q' "$CACHE_HOST_PATH") JIT_CACHE_HOST_PATH=$(printf '%q' "$JIT_CACHE_HOST_PATH") ENFORCE_EAGER=$(printf '%q' "$ENFORCE_EAGER") COMPILATION_CONFIG=$(printf '%q' "$COMPILATION_CONFIG") ENABLE_DFLASH=$(printf '%q' "$ENABLE_DFLASH") DFLASH_TOKENS=$(printf '%q' "$DFLASH_TOKENS") DFLASH_DRAFT_TP=$(printf '%q' "$DFLASH_DRAFT_TP") DFLASH_DRAFT_SAMPLE_METHOD=$(printf '%q' "$DFLASH_DRAFT_SAMPLE_METHOD") DFLASH_REJECTION_SAMPLE_METHOD=$(printf '%q' "$DFLASH_REJECTION_SAMPLE_METHOD") EXL3_FUSED_MOE=$(printf '%q' "$EXL3_FUSED_MOE") EXL3_MOE_ROW_TILE=$(printf '%q' "$EXL3_MOE_ROW_TILE") EXL3_FAT_KERNEL=$(printf '%q' "$EXL3_FAT_KERNEL") EXL3_TEMP_ROWS_FUSED=$(printf '%q' "$EXL3_TEMP_ROWS_FUSED") COMPACT_SPEC_REPLAY=$(printf '%q' "$COMPACT_SPEC_REPLAY") GLM53_SPINWAIT_MS=$(printf '%q' "$GLM53_SPINWAIT_MS") VLLM_B12X_USE_CUDA_GRAPH=$(printf '%q' "$VLLM_B12X_USE_CUDA_GRAPH") VLLM_B12X_CUDA_GRAPH_MAX_TOKENS=$(printf '%q' "$VLLM_B12X_CUDA_GRAPH_MAX_TOKENS") ENABLE_DECODE_FIRST_SCHEDULER=$(printf '%q' "$ENABLE_DECODE_FIRST_SCHEDULER") PREFILL_ADMISSION_POLICY=$(printf '%q' "$PREFILL_ADMISSION_POLICY") PREFILL_SCHEDULE_INTERVAL=$(printf '%q' "$PREFILL_SCHEDULE_INTERVAL") LONG_PREFILL_TOKEN_THRESHOLD=$(printf '%q' "$LONG_PREFILL_TOKEN_THRESHOLD") ADAPTIVE_SCHEDULER_CONFIG=$(printf '%q' "$worker_adaptive_scheduler_config") ADAPTIVE_SCHEDULER_RELOAD_SECONDS=$(printf '%q' "$ADAPTIVE_SCHEDULER_RELOAD_SECONDS") NVFP4_CALIBRATION_HOST_PATH=$(printf '%q' "$worker_calibration_host_path") NVFP4_MLA_SCALES_FILE=$(printf '%q' "$NVFP4_MLA_SCALES_FILE") ENABLE_NVFP4_MLA_CAPTURE=$(printf '%q' "$ENABLE_NVFP4_MLA_CAPTURE") NVFP4_MLA_CAPTURE_GROUPS_PER_STRATUM=$(printf '%q' "$NVFP4_MLA_CAPTURE_GROUPS_PER_STRATUM") NVFP4_MLA_CAPTURE_GROUPS_PER_CALL=$(printf '%q' "$NVFP4_MLA_CAPTURE_GROUPS_PER_CALL") ./launch-glm53-vllm-tp2-dflash2.sh 1"
sleep 25
echo "Starting head rank on $(hostname)..."
CONTAINER_NAME="$CONTAINER_NAME" IMAGE="$IMAGE" MODEL_HOST_PATH="$MODEL_HOST_PATH" API_PORT="$API_PORT" \
  DRAFT_HOST_PATH="$DRAFT_HOST_PATH" \
  HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" \
  DCP_SIZE="$DCP_SIZE" USE_FP4_INDEXER_CACHE="$USE_FP4_INDEXER_CACHE" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  BLOCK_SIZE="$BLOCK_SIZE" \
  CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING" \
  QUANTIZATION="$QUANTIZATION" KV_CACHE_DTYPE="$KV_CACHE_DTYPE" \
  VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND" \
  CACHE_HOST_PATH="$CACHE_HOST_PATH" JIT_CACHE_HOST_PATH="$JIT_CACHE_HOST_PATH" \
  ENABLE_DFLASH="$ENABLE_DFLASH" DFLASH_TOKENS="$DFLASH_TOKENS" \
  DFLASH_DRAFT_TP="$DFLASH_DRAFT_TP" \
  DFLASH_DRAFT_SAMPLE_METHOD="$DFLASH_DRAFT_SAMPLE_METHOD" \
  DFLASH_REJECTION_SAMPLE_METHOD="$DFLASH_REJECTION_SAMPLE_METHOD" \
  EXL3_FUSED_MOE="$EXL3_FUSED_MOE" EXL3_MOE_ROW_TILE="$EXL3_MOE_ROW_TILE" \
  EXL3_FAT_KERNEL="$EXL3_FAT_KERNEL" EXL3_TEMP_ROWS_FUSED="$EXL3_TEMP_ROWS_FUSED" \
  COMPACT_SPEC_REPLAY="$COMPACT_SPEC_REPLAY" GLM53_SPINWAIT_MS="$GLM53_SPINWAIT_MS" \
  ENABLE_DECODE_FIRST_SCHEDULER="$ENABLE_DECODE_FIRST_SCHEDULER" \
  PREFILL_ADMISSION_POLICY="$PREFILL_ADMISSION_POLICY" \
  MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" MOE_BACKEND="$MOE_BACKEND" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  COMPILATION_CONFIG="$COMPILATION_CONFIG" \
  ENFORCE_EAGER="$ENFORCE_EAGER" VLLM_B12X_USE_CUDA_GRAPH="$VLLM_B12X_USE_CUDA_GRAPH" \
  VLLM_B12X_CUDA_GRAPH_MAX_TOKENS="$VLLM_B12X_CUDA_GRAPH_MAX_TOKENS" \
  PREFILL_SCHEDULE_INTERVAL="$PREFILL_SCHEDULE_INTERVAL" \
  LONG_PREFILL_TOKEN_THRESHOLD="$LONG_PREFILL_TOKEN_THRESHOLD" \
  ADAPTIVE_SCHEDULER_CONFIG="$ADAPTIVE_SCHEDULER_CONFIG" \
  ADAPTIVE_SCHEDULER_RELOAD_SECONDS="$ADAPTIVE_SCHEDULER_RELOAD_SECONDS" \
  NVFP4_CALIBRATION_HOST_PATH="$NVFP4_CALIBRATION_HOST_PATH" \
  NVFP4_MLA_SCALES_FILE="$NVFP4_MLA_SCALES_FILE" \
  ENABLE_NVFP4_MLA_CAPTURE="$ENABLE_NVFP4_MLA_CAPTURE" \
  NVFP4_MLA_CAPTURE_GROUPS_PER_STRATUM="$NVFP4_MLA_CAPTURE_GROUPS_PER_STRATUM" \
  NVFP4_MLA_CAPTURE_GROUPS_PER_CALL="$NVFP4_MLA_CAPTURE_GROUPS_PER_CALL" \
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
