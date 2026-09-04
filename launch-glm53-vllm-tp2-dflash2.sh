#!/usr/bin/env bash
#
# GLM-5.3-Flash + DFlash2 speculative decoding, TP2/DCP2 on 2x DGX Spark.
# Defaults select the validated compact native-FP4 KV-cache image.
#
# Prerequisites (see README Quickstart):
#   1. image glm53-v13:nvfp4-four-over-six present on BOTH nodes (or set IMAGE)
#   2. weights at $MODEL_HOST_PATH on BOTH nodes
#   3. drafter (2.2 GB) at /var/tmp/models/GLM-5.3-Flash-DFlash2 on BOTH nodes
#   4. cp docker/sparse_attn_indexer_kpool_sm121.py $HOME/patches/sparse_attn_indexer_kpool.py
#      on BOTH nodes -- the SM121 top-k fix. Without it the engine dies on any
#      decode past ~24K context (docs/SM121-CRASH-FORENSICS-2026-08-27.md).
#
# Usage: ./launch-glm53-vllm-tp2-dflash2.sh <0|1>   -- worker (1) FIRST, then head (0)
set -euo pipefail

# Run worker first, wait briefly, then run the head. start-cluster.sh does this
# automatically; this lower-level launcher remains useful for diagnostics.
NODE_RANK="${1:?usage: launch-glm53-vllm-tp2-dflash2.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-glm53-v13:nvfp4-four-over-six}"
NAME="${CONTAINER_NAME:-vllm_glm53}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-$HOME/.cache/huggingface/glm53-redhat-nvfp4-fp8-passthrough-v1}"
MODEL_PATH="/models/glm-5.3-flash-nvfp4"
CACHE_HOST_PATH="${CACHE_HOST_PATH:-$HOME/.cache/huggingface/vllm-cache-glm53}"
JIT_CACHE_HOST_PATH="${JIT_CACHE_HOST_PATH:-$HOME/.cache/glm53-vllm-jit}"
DRAFT_HOST_PATH="${DRAFT_HOST_PATH:-$HOME/.cache/huggingface/glm53-dflash2-bf582e4eacc1810f76656d1811693ff6c6737d2a}"
TOPK_PATCH="${TOPK_PATCH:-$SCRIPT_DIR/docker/sparse_attn_indexer_kpool_sm121.py}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/chat_template_mm.jinja}"
SCHEDULER_ADAPTER="${SCHEDULER_ADAPTER:-$SCRIPT_DIR/docker/glm_decode_first_scheduler.py}"
HEAD_IP="${HEAD_IP:-10.100.32.1}"
WORKER_IP="${WORKER_IP:-10.100.32.2}"
MPORT="${MASTER_PORT:-29521}"
PORT="${API_PORT:-8000}"
NCCL_HCA="${NCCL_IB_HCA:-rocep1s0f1}"
NCCL_IFACE="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
NCCL_SUBNET="${NCCL_IB_ADDR_RANGE:-10.100.32.0/24}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
BLOCK_SIZE="${BLOCK_SIZE:-2304}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.87}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-}"
VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-}"
VLLM_DISABLE_FLASHINFER_AUTOTUNE="${VLLM_DISABLE_FLASHINFER_AUTOTUNE:-0}"
USE_FP4_INDEXER_CACHE="${USE_FP4_INDEXER_CACHE:-0}"
DCP_SIZE="${DCP_SIZE:-2}"
ENABLE_DFLASH="${ENABLE_DFLASH:-1}"
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
DFLASH_DRAFT_VARIANT="${DFLASH_DRAFT_VARIANT-}"
DFLASH_DRAFT_QUANTIZATION="${DFLASH_DRAFT_QUANTIZATION-}"
VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM-}"
if [[ -z "$DFLASH_DRAFT_VARIANT" ]]; then
  if [[ -f "$DRAFT_HOST_PATH/hf_quant_config.json" ]]; then
    DFLASH_DRAFT_VARIANT=mxfp8
  else
    DFLASH_DRAFT_VARIANT=bf16
  fi
fi
if [[ "$DFLASH_DRAFT_VARIANT" == "mxfp8" ]]; then
  : "${DFLASH_DRAFT_QUANTIZATION:=modelopt_mxfp8}"
  : "${VLLM_USE_B12X_FP8_GEMM:=1}"
else
  : "${VLLM_USE_B12X_FP8_GEMM:=0}"
fi
DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP:-2}"
DFLASH_DRAFT_SAMPLE_METHOD="${DFLASH_DRAFT_SAMPLE_METHOD-}"
DFLASH_REJECTION_SAMPLE_METHOD="${DFLASH_REJECTION_SAMPLE_METHOD-}"
QUANTIZATION="${QUANTIZATION:-}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
COMPILATION_CONFIG="${COMPILATION_CONFIG-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
ENABLE_DECODE_FIRST_SCHEDULER="${ENABLE_DECODE_FIRST_SCHEDULER:-1}"
PREFILL_ADMISSION_POLICY="${PREFILL_ADMISSION_POLICY:-adaptive}"
PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-4}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-512}"
ADAPTIVE_SCHEDULER_CONFIG="${ADAPTIVE_SCHEDULER_CONFIG:-$SCRIPT_DIR/scheduler_profiles/adaptive.json}"
ADAPTIVE_SCHEDULER_RELOAD_SECONDS="${ADAPTIVE_SCHEDULER_RELOAD_SECONDS:-1}"
EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"
EXL3_MOE_ROW_TILE="${EXL3_MOE_ROW_TILE:-0}"
EXL3_FAT_KERNEL="${EXL3_FAT_KERNEL:-1}"
EXL3_TEMP_ROWS_FUSED="${EXL3_TEMP_ROWS_FUSED:-128}"
COMPACT_SPEC_REPLAY="${COMPACT_SPEC_REPLAY:-1}"
GLM53_SPINWAIT_MS="${GLM53_SPINWAIT_MS:-stock}"
VLLM_B12X_USE_CUDA_GRAPH="${VLLM_B12X_USE_CUDA_GRAPH:-0}"
VLLM_B12X_CUDA_GRAPH_MAX_TOKENS="${VLLM_B12X_CUDA_GRAPH_MAX_TOKENS:-64}"
VLLM_DFLASH_ONLY_CUDAGRAPH="${VLLM_DFLASH_ONLY_CUDAGRAPH:-0}"
VLLM_DFLASH_CUDAGRAPH_BATCHES="${VLLM_DFLASH_CUDAGRAPH_BATCHES:-1,2,3,4,5,6}"
NVFP4_CALIBRATION_HOST_PATH="${NVFP4_CALIBRATION_HOST_PATH:-}"
NVFP4_MLA_SCALES_FILE="${NVFP4_MLA_SCALES_FILE:-}"
ENABLE_NVFP4_MLA_CAPTURE="${ENABLE_NVFP4_MLA_CAPTURE:-0}"
NVFP4_MLA_CAPTURE_GROUPS_PER_STRATUM="${NVFP4_MLA_CAPTURE_GROUPS_PER_STRATUM:-65536}"
NVFP4_MLA_CAPTURE_GROUPS_PER_CALL="${NVFP4_MLA_CAPTURE_GROUPS_PER_CALL:-1024}"
attention_backend_args=()
if [[ -n "$VLLM_ATTENTION_BACKEND" ]]; then
  attention_backend_args+=( --attention-backend "$VLLM_ATTENTION_BACKEND" )
fi
load_format_args=()
if [[ -n "$VLLM_LOAD_FORMAT" ]]; then
  load_format_args+=( --load-format "$VLLM_LOAD_FORMAT" )
fi
flashinfer_autotune_args=()
if [[ "$VLLM_DISABLE_FLASHINFER_AUTOTUNE" == "1" ]]; then
  flashinfer_autotune_args+=( --no-enable-flashinfer-autotune )
fi
attention_config_args=()
if [[ "$USE_FP4_INDEXER_CACHE" == "1" ]]; then
  attention_config_args+=( --attention-config '{"use_fp4_indexer_cache":true}' )
fi
dcp_args=()
if [[ "$DCP_SIZE" != "1" ]]; then
  dcp_args+=( --decode-context-parallel-size "$DCP_SIZE" )
fi
speculative_args=()
if [[ "$ENABLE_DFLASH" == "1" ]]; then
  [[ "$DFLASH_TOKENS" =~ ^[1-8]$ ]] || {
    echo "DFLASH_TOKENS must be an integer from 1 through 8" >&2
    exit 2
  }
  [[ -z "$DFLASH_DRAFT_TP" || "$DFLASH_DRAFT_TP" == "1" || "$DFLASH_DRAFT_TP" == "2" ]] || {
    echo "DFLASH_DRAFT_TP must be empty, 1, or 2" >&2
    exit 2
  }
  [[ -z "$DFLASH_DRAFT_SAMPLE_METHOD" || "$DFLASH_DRAFT_SAMPLE_METHOD" == "probabilistic" ]] || {
    echo "DFLASH_DRAFT_SAMPLE_METHOD must be empty or probabilistic" >&2
    exit 2
  }
  [[ -z "$DFLASH_REJECTION_SAMPLE_METHOD" || "$DFLASH_REJECTION_SAMPLE_METHOD" == "standard" ]] || {
    echo "DFLASH_REJECTION_SAMPLE_METHOD must be empty or standard" >&2
    exit 2
  }
  [[ "$DFLASH_DRAFT_VARIANT" == "bf16" || "$DFLASH_DRAFT_VARIANT" == "mxfp8" ]] || {
    echo "DFLASH_DRAFT_VARIANT must be bf16 or mxfp8" >&2
    exit 2
  }
  speculative_config='{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":'
  speculative_config+="$DFLASH_TOKENS"
  if [[ -n "$DFLASH_DRAFT_TP" ]]; then
    speculative_config+=',"draft_tensor_parallel_size":'
    speculative_config+="$DFLASH_DRAFT_TP"
  fi
  if [[ -n "$DFLASH_DRAFT_QUANTIZATION" ]]; then
    speculative_config+=',"quantization":"'"$DFLASH_DRAFT_QUANTIZATION"'"'
  fi
  if [[ -n "$DFLASH_DRAFT_SAMPLE_METHOD" ]]; then
    speculative_config+=',"draft_sample_method":"probabilistic"'
  fi
  if [[ -n "$DFLASH_REJECTION_SAMPLE_METHOD" ]]; then
    speculative_config+=',"rejection_sample_method":"standard"'
  fi
  speculative_config+='}'
  speculative_args+=( --speculative-config "$speculative_config" )
fi
quantization_args=()
if [[ -n "$QUANTIZATION" ]]; then
  quantization_args+=( --quantization "$QUANTIZATION" )
fi
moe_backend_args=()
if [[ -n "$MOE_BACKEND" && "$MOE_BACKEND" != "auto" ]]; then
  moe_backend_args+=( --moe-backend "$MOE_BACKEND" )
fi
eager_args=()
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  eager_args+=( --enforce-eager )
fi
compilation_args=()
if [[ -n "$COMPILATION_CONFIG" ]]; then
  python3 -c 'import json,sys; json.loads(sys.argv[1])' "$COMPILATION_CONFIG"
  compilation_args+=( --compilation-config "$COMPILATION_CONFIG" )
fi
profiler_args=()
if [[ -n "${PROFILER_CONFIG:-}" ]]; then
  profiler_args+=( --profiler-config "$PROFILER_CONFIG" )
fi
scheduler_mount_args=()
scheduler_env_args=()
scheduler_args=()
case "$ENABLE_DECODE_FIRST_SCHEDULER" in
  0) ;;
  1)
    case "$PREFILL_ADMISSION_POLICY" in
      static)
        scheduler_class="glm_decode_first_scheduler.DecodeFirstScheduler"
        ;;
      adaptive)
        scheduler_class="glm_decode_first_scheduler.AdaptiveDecodeFirstScheduler"
        test -s "$ADAPTIVE_SCHEDULER_CONFIG"
        python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$ADAPTIVE_SCHEDULER_CONFIG"
        adaptive_config_dir="$(
          cd -- "$(dirname -- "$ADAPTIVE_SCHEDULER_CONFIG")" && pwd
        )"
        adaptive_config_name="$(basename -- "$ADAPTIVE_SCHEDULER_CONFIG")"
        scheduler_mount_args+=(
          -v "$adaptive_config_dir:/etc/glm53/adaptive-scheduler:ro"
        )
        scheduler_env_args+=(
          -e "GLM_ADAPTIVE_PREFILL_CONFIG=/etc/glm53/adaptive-scheduler/$adaptive_config_name"
          -e "GLM_ADAPTIVE_PREFILL_RELOAD_SECONDS=$ADAPTIVE_SCHEDULER_RELOAD_SECONDS"
        )
        ;;
      *)
        echo "PREFILL_ADMISSION_POLICY must be static or adaptive" >&2
        exit 2
        ;;
    esac
    [[ "$PREFILL_SCHEDULE_INTERVAL" =~ ^[1-9][0-9]*$ ]] || {
      echo "PREFILL_SCHEDULE_INTERVAL must be a positive integer" >&2
      exit 2
    }
    [[ "$LONG_PREFILL_TOKEN_THRESHOLD" =~ ^[0-9]+$ ]] || {
      echo "LONG_PREFILL_TOKEN_THRESHOLD must be a non-negative integer" >&2
      exit 2
    }
    test -s "$SCHEDULER_ADAPTER"
    scheduler_mount_args+=(
      -v "$SCHEDULER_ADAPTER:/usr/local/lib/python3.12/dist-packages/glm_decode_first_scheduler.py:ro"
    )
    scheduler_args+=(
      --scheduler-cls "$scheduler_class"
      --prefill-schedule-interval "$PREFILL_SCHEDULE_INTERVAL"
      --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD"
    )
    ;;
  *)
    echo "ENABLE_DECODE_FIRST_SCHEDULER must be 0 or 1" >&2
    exit 2
    ;;
esac


calibration_docker_args=()
if [[ -n "$NVFP4_MLA_SCALES_FILE" || "$ENABLE_NVFP4_MLA_CAPTURE" == "1" ]]; then
  [[ -n "$NVFP4_CALIBRATION_HOST_PATH" ]] || {
    echo "NVFP4_CALIBRATION_HOST_PATH is required for scales or capture" >&2
    exit 2
  }
  mkdir -p "$NVFP4_CALIBRATION_HOST_PATH"
  calibration_docker_args+=(
    -v "$NVFP4_CALIBRATION_HOST_PATH:/calibration"
  )
fi
if [[ -n "$NVFP4_MLA_SCALES_FILE" ]]; then
  [[ "$NVFP4_MLA_SCALES_FILE" != */* ]] || {
    echo "NVFP4_MLA_SCALES_FILE must be a filename inside the calibration directory" >&2
    exit 2
  }
  test -s "$NVFP4_CALIBRATION_HOST_PATH/$NVFP4_MLA_SCALES_FILE"
  calibration_docker_args+=(
    -e "VLLM_NVFP4_MLA_SCALES_FILE=/calibration/$NVFP4_MLA_SCALES_FILE"
    -e VLLM_NVFP4_MLA_SCALES_STRICT=1
  )
fi
case "$ENABLE_NVFP4_MLA_CAPTURE" in
  0) ;;
  1)
    calibration_docker_args+=(
      -e VLLM_NVFP4_MLA_CAPTURE_DIR=/calibration/captures
      -e VLLM_NVFP4_MLA_CAPTURE_CONTROL=/calibration/control.json
      -e "VLLM_NVFP4_MLA_CAPTURE_GROUPS_PER_STRATUM=$NVFP4_MLA_CAPTURE_GROUPS_PER_STRATUM"
      -e "VLLM_NVFP4_MLA_CAPTURE_GROUPS_PER_CALL=$NVFP4_MLA_CAPTURE_GROUPS_PER_CALL"
    )
    ;;
  *)
    echo "ENABLE_NVFP4_MLA_CAPTURE must be 0 or 1" >&2
    exit 2
    ;;
esac

case "$NODE_RANK" in
  0) HOST_IP="$HEAD_IP"; HEADLESS="" ;;
  1) HOST_IP="$WORKER_IP"; HEADLESS="--headless" ;;
esac

test -f "$MODEL_HOST_PATH/config.json"
test -f "$MODEL_HOST_PATH/model.safetensors.index.json"
test -f "$DRAFT_HOST_PATH/config.json"
test -f "$DRAFT_HOST_PATH/model.safetensors"
if [[ "$DFLASH_DRAFT_VARIANT" == "mxfp8" ]]; then
  test -f "$DRAFT_HOST_PATH/hf_quant_config.json"
  test -f "$DRAFT_HOST_PATH/conversion_manifest.json"
  python3 -c 'import json,sys
q=json.load(open(sys.argv[1]))
algo=(q.get("quantization") or {}).get("quant_algo", q.get("quant_algo"))
assert algo == "MXFP8", f"expected MXFP8 draft, got {algo!r}"' \
    "$DRAFT_HOST_PATH/hf_quant_config.json"
  [[ "$DFLASH_DRAFT_QUANTIZATION" == "modelopt_mxfp8" ]] || {
    echo "MXFP8 draft requires DFLASH_DRAFT_QUANTIZATION=modelopt_mxfp8" >&2
    exit 2
  }
fi
test -s "$TOPK_PATCH"
test -s "$CHAT_TEMPLATE"
mkdir -p "$CACHE_HOST_PATH" "$JIT_CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -v "$JIT_CACHE_HOST_PATH:/root/.cache" \
  -e VLLM_HOST_IP=$HOST_IP \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$NCCL_HCA" -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE="$NCCL_SUBNET" \
  -e NCCL_SOCKET_IFNAME="$NCCL_IFACE" -e GLOO_SOCKET_IFNAME="$NCCL_IFACE" \
  -e TP_SOCKET_IFNAME="$NCCL_IFACE" -e MN_IF_NAME="$NCCL_IFACE" \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "${scheduler_mount_args[@]}" \
  "${scheduler_env_args[@]}" \
  -v "$TOPK_PATCH:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro" \
  -v "$DRAFT_HOST_PATH:/models/dflash2-draft:ro" \
  -e EXL3_FUSED_MOE="$EXL3_FUSED_MOE" \
  -e EXL3_MOE_ROW_TILE="$EXL3_MOE_ROW_TILE" \
  -e EXL3_FAT_KERNEL="$EXL3_FAT_KERNEL" \
  -e EXL3_FUSED_FAT_ACTIVATION="${EXL3_FUSED_FAT_ACTIVATION:-0}" \
  -e GLM53_EXL3_MOE_FAST="${GLM53_EXL3_MOE_FAST:-0}" \
  -e EXL3_FAT_ACTIVATION_CONTROL="${EXL3_FAT_ACTIVATION_CONTROL:-}" \
  -e EXL3_TEMP_ROWS_FUSED="$EXL3_TEMP_ROWS_FUSED" \
  -e EXL3_FAT_SCRATCH_ROWS="$MAX_NUM_BATCHED_TOKENS" \
  -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  -e VLLM_COMPACT_SPEC_REPLAY="$COMPACT_SPEC_REPLAY" \
  -e GLM53_SPINWAIT_MS="$GLM53_SPINWAIT_MS" \
  -e VLLM_B12X_USE_CUDA_GRAPH="$VLLM_B12X_USE_CUDA_GRAPH" \
  -e VLLM_B12X_CUDA_GRAPH_MAX_TOKENS="$VLLM_B12X_CUDA_GRAPH_MAX_TOKENS" \
  -e VLLM_DFLASH_ONLY_CUDAGRAPH="$VLLM_DFLASH_ONLY_CUDAGRAPH" \
  -e VLLM_DFLASH_CUDAGRAPH_BATCHES="$VLLM_DFLASH_CUDAGRAPH_BATCHES" \
  -e VLLM_USE_B12X_FP8_GEMM="$VLLM_USE_B12X_FP8_GEMM" \
  -v "$CHAT_TEMPLATE:/models/chat_template_mm.jinja:ro" \
  "${calibration_docker_args[@]}" \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    "${attention_backend_args[@]}" \
    "${load_format_args[@]}" \
    "${flashinfer_autotune_args[@]}" \
    "${quantization_args[@]}" \
    "${attention_config_args[@]}" \
    --tensor-parallel-size 2 \
    "${dcp_args[@]}" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" --block-size "$BLOCK_SIZE" "${moe_backend_args[@]}" "${speculative_args[@]}" --kv-cache-dtype "$KV_CACHE_DTYPE" \
    "${eager_args[@]}" "${compilation_args[@]}" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    "${scheduler_args[@]}" "${profiler_args[@]}" \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"high"}' --chat-template /models/chat_template_mm.jinja \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited; inspect with: docker logs $NAME" >&2
  exit 1
}
