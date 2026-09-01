#!/usr/bin/env bash
# GLM-5.3-Flash EXL3 + 528-byte FP8 MLA + DFlash2, TP2/DCP2.
# Start rank 1 first, then rank 0. start-exl3-fp8-dcp2-cluster.sh automates it.
set -Eeuo pipefail

NODE_RANK="${1:?usage: launch-glm53-exl3-fp8-dcp2.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || {
  echo "rank must be 0 or 1" >&2
  exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-glm53-exl3:fp8-dcp2}"
NAME="${CONTAINER_NAME:-vllm_glm53}"
MODEL_REVISION="25a44fdbf16862a46b7cc9921142c6c81350af2f"
DRAFT_REVISION="dc77ff1c99eeb2df044ee3d4f0094eb033fee410"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-$HOME/.cache/huggingface/glm53-exl3-tr3-4bpw-$MODEL_REVISION}"
DRAFT_HOST_PATH="${DRAFT_HOST_PATH:-$HOME/.cache/huggingface/glm53-dflash2-$DRAFT_REVISION}"
MODEL_PATH="/models/glm-5.3-flash-exl3"
DRAFT_PATH="/models/dflash2-draft"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/chat_template_mm.jinja}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/glm53-exl3-fp8-dcp2}"
HEAD_IP="${HEAD_IP:-10.100.32.1}"
WORKER_IP="${WORKER_IP:-10.100.32.2}"
MPORT="${MASTER_PORT:-29521}"
PORT="${API_PORT:-8000}"
NCCL_HCA="${NCCL_IB_HCA:-rocep1s0f1}"
NCCL_IFACE="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
NCCL_SUBNET="${NCCL_IB_ADDR_RANGE:-10.100.32.0/24}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.87}"
ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER_MLA_SPARSE_SM120}"
DCP_SIZE="${DCP_SIZE:-2}"
ENABLE_DFLASH="${ENABLE_DFLASH:-1}"
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP:-1}"
COMPACT_SPEC_REPLAY="${COMPACT_SPEC_REPLAY:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

[[ "$DCP_SIZE" == "2" ]] || {
  echo "This profile is specifically validated for DCP_SIZE=2" >&2
  exit 2
}
[[ "$ENFORCE_EAGER" == "0" || "$ENFORCE_EAGER" == "1" ]] || {
  echo "ENFORCE_EAGER must be 0 or 1" >&2
  exit 2
}
[[ "$DFLASH_TOKENS" =~ ^[1-7]$ ]] || {
  echo "DFLASH_TOKENS must be an integer from 1 through 7" >&2
  exit 2
}
[[ "$DFLASH_DRAFT_TP" == "1" || "$DFLASH_DRAFT_TP" == "2" ]] || {
  echo "DFLASH_DRAFT_TP must be 1 or 2" >&2
  exit 2
}
[[ "$COMPACT_SPEC_REPLAY" == "0" || "$COMPACT_SPEC_REPLAY" == "1" ]] || {
  echo "COMPACT_SPEC_REPLAY must be 0 or 1" >&2
  exit 2
}
case "$NODE_RANK" in
  0) HOST_IP="$HEAD_IP"; HEADLESS=() ;;
  1) HOST_IP="$WORKER_IP"; HEADLESS=(--headless) ;;
esac

test -f "$MODEL_HOST_PATH/config.json"
test -f "$MODEL_HOST_PATH/model.safetensors.index.json"
test -f "$DRAFT_HOST_PATH/config.json"
test -f "$DRAFT_HOST_PATH/model.safetensors"
test -s "$CHAT_TEMPLATE"
mkdir -p "$CACHE_ROOT/vllm" "$CACHE_ROOT/triton" "$CACHE_ROOT/tilelang"

speculative_args=()
if [[ "$ENABLE_DFLASH" == "1" ]]; then
  speculative_config='{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":'
  speculative_config+="$DFLASH_TOKENS"
  speculative_config+=',"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","draft_tensor_parallel_size":'
  speculative_config+="$DFLASH_DRAFT_TP"
  speculative_config+='}'
  speculative_args+=(
    --speculative-config
    "$speculative_config"
  )
fi

execution_args=(--cudagraph-capture-sizes 1 2 4 8 16 24 32)
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  execution_args=(--enforce-eager)
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v "$DRAFT_HOST_PATH:$DRAFT_PATH:ro" \
  -v "$CHAT_TEMPLATE:/models/chat_template_mm.jinja:ro" \
  -v "$CACHE_ROOT/vllm:/root/.cache/vllm" \
  -v "$CACHE_ROOT/triton:/root/.triton/cache" \
  -v "$CACHE_ROOT/tilelang:/root/.tilelang/cache" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e VLLM_COMPACT_SPEC_REPLAY="$COMPACT_SPEC_REPLAY" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}" \
  -e EXL3_MOE_ROW_TILE="${EXL3_MOE_ROW_TILE:-0}" \
  -e EXL3_TEMP_ROWS_FUSED="${EXL3_TEMP_ROWS_FUSED:-128}" \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$NCCL_HCA" -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE="$NCCL_SUBNET" \
  -e NCCL_SOCKET_IFNAME="$NCCL_IFACE" -e GLOO_SOCKET_IFNAME="$NCCL_IFACE" \
  -e TP_SOCKET_IFNAME="$NCCL_IFACE" -e MN_IF_NAME="$NCCL_IFACE" \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --quantization exl3 \
    --attention-backend "$ATTENTION_BACKEND" \
    --attention-config '{"use_fp4_indexer_cache":true}' \
    --tensor-parallel-size 2 \
    --decode-context-parallel-size 2 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --block-size 2304 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching \
    --no-enable-flashinfer-autotune \
    "${execution_args[@]}" \
    "${speculative_args[@]}" \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --chat-template /models/chat_template_mm.jinja \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    "${HEADLESS[@]}"

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep -F "$NAME" || {
  echo "$NAME exited; inspect with: docker logs $NAME" >&2
  exit 1
}
