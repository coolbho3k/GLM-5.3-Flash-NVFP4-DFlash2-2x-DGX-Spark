#!/usr/bin/env bash
# Select a reproducible GLM-5.3 weight/KV profile while sharing serving knobs.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXL3_MODEL_REVISION="25a44fdbf16862a46b7cc9921142c6c81350af2f"
NVFP4_MODEL_DEFAULT="$HOME/.cache/huggingface/glm53-nvfp4-marlin-shared-w13-v1"
EXL3_MODEL_DEFAULT="$HOME/.cache/huggingface/glm53-exl3-tr3-4bpw-$EXL3_MODEL_REVISION"
DRAFT_DEFAULT="$HOME/.cache/huggingface/glm53-dflash2-bf582e4eacc1810f76656d1811693ff6c6737d2a"

usage() {
  cat <<'EOF'
Usage:
  ./serve-profile.sh list
  ./serve-profile.sh show  PROFILE
  ./serve-profile.sh start PROFILE
  ./serve-profile.sh build PROFILE
  ./serve-profile.sh prepare PROFILE
  ./serve-profile.sh stop

Profiles:
  nvfp4-fp4-dcp2  Current production: optimized NVFP4 weights + calibrated native-FP4 KV
  nvfp4-fp8-dcp2  Optimized NVFP4 weights + compact NoPE FP8 KV
  exl3-fp4-dcp2   EXL3 weights + calibrated native-FP4 KV
  exl3-fp8-dcp2   EXL3 weights + compact NoPE FP8 KV

Every setting is an environment override. For example:
  MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048 \
    ./serve-profile.sh start exl3-fp8-dcp2
  DCP_SIZE=1 ENABLE_DFLASH=0 ./serve-profile.sh start exl3-fp8-dcp2

Piecewise CUDA graphs are retained for diagnostics but are slower here:
  ENFORCE_EAGER=0 \
  COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"PIECEWISE"}' \
    ./serve-profile.sh start exl3-fp8-dcp2
EOF
}

default_var() {
  local name="$1" value="$2"
  if [[ ! -v "$name" ]]; then
    printf -v "$name" '%s' "$value"
    export "$name"
  fi
}

list_profiles() {
  printf '%-19s  %-7s  %-10s  %-4s  %s\n' PROFILE WEIGHTS TARGET_KV DCP STATUS
  printf '%-19s  %-7s  %-10s  %-4s  %s\n' nvfp4-fp4-dcp2 NVFP4 native-FP4 2 validated
  printf '%-19s  %-7s  %-10s  %-4s  %s\n' nvfp4-fp8-dcp2 NVFP4 FP8-NoPE 2 runnable-512K
  printf '%-19s  %-7s  %-10s  %-4s  %s\n' exl3-fp4-dcp2 EXL3 native-FP4 2 validated-1M
  printf '%-19s  %-7s  %-10s  %-4s  %s\n' exl3-fp8-dcp2 EXL3 FP8-NoPE 2 validated-1M
}

configure_profile() {
  local profile="$1"
  case "$profile" in
    nvfp4-fp4-dcp2)
      default_var IMAGE glm53-v14:nvfp4-gscale-tooling
      default_var MODEL_HOST_PATH "$NVFP4_MODEL_DEFAULT"
      default_var QUANTIZATION ""
      default_var MOE_BACKEND marlin
      default_var VLLM_ATTENTION_BACKEND ""
      default_var USE_CALIBRATED_NVFP4_MLA 1
      default_var SYNC_IMAGE_TO_WORKER 0
      ;;
    nvfp4-fp8-dcp2)
      default_var IMAGE glm53-exl3:e2-fp8-dcp2
      default_var MODEL_HOST_PATH "$NVFP4_MODEL_DEFAULT"
      default_var QUANTIZATION ""
      default_var MOE_BACKEND marlin
      default_var VLLM_ATTENTION_BACKEND FLASHINFER_MLA_SPARSE_SM120
      default_var USE_CALIBRATED_NVFP4_MLA 0
      default_var BLOCK_SIZE 2048
      # The corrected Marlin checkpoint plus an 8192-token profiling batch
      # leaves 6.99 GiB for FP8 KV at GMU=0.87. That is enough for ~620K
      # logical tokens, but below vLLM's 11.8-GiB admission floor for 1M.
      default_var MAX_MODEL_LEN 524288
      default_var SYNC_IMAGE_TO_WORKER 1
      ;;
    exl3-fp4-dcp2)
      default_var IMAGE glm53-exl3:e2-native-fp4-dcp2
      default_var MODEL_HOST_PATH "$EXL3_MODEL_DEFAULT"
      default_var QUANTIZATION exl3
      default_var MOE_BACKEND auto
      default_var VLLM_ATTENTION_BACKEND ""
      default_var USE_CALIBRATED_NVFP4_MLA 1
      default_var SYNC_IMAGE_TO_WORKER 1
      ;;
    exl3-fp8-dcp2)
      default_var IMAGE glm53-exl3:e2-fp8-dcp2
      default_var MODEL_HOST_PATH "$EXL3_MODEL_DEFAULT"
      default_var QUANTIZATION exl3
      default_var MOE_BACKEND auto
      default_var VLLM_ATTENTION_BACKEND FLASHINFER_MLA_SPARSE_SM120
      default_var USE_CALIBRATED_NVFP4_MLA 0
      default_var BLOCK_SIZE 2048
      default_var SYNC_IMAGE_TO_WORKER 1
      ;;
    *)
      echo "Unknown profile: $profile" >&2
      list_profiles >&2
      exit 2
      ;;
  esac

  default_var DRAFT_HOST_PATH "$DRAFT_DEFAULT"
  default_var DCP_SIZE 2
  default_var USE_FP4_INDEXER_CACHE 0
  default_var BLOCK_SIZE 2304
  default_var GPU_MEMORY_UTILIZATION 0.87
  default_var MAX_MODEL_LEN 1048576
  default_var MAX_NUM_SEQS 6
  default_var MAX_NUM_BATCHED_TOKENS 8192
  default_var ENFORCE_EAGER 1
  default_var COMPILATION_CONFIG ""
  default_var ENABLE_DFLASH 1
  default_var DFLASH_TOKENS 7
  default_var DFLASH_DRAFT_TP ""
  default_var DFLASH_DRAFT_SAMPLE_METHOD ""
  default_var DFLASH_REJECTION_SAMPLE_METHOD ""
  default_var ENABLE_DECODE_FIRST_SCHEDULER 1
  default_var PREFILL_ADMISSION_POLICY adaptive
  default_var PREFILL_SCHEDULE_INTERVAL 4
  default_var LONG_PREFILL_TOKEN_THRESHOLD 512
  default_var EXL3_FUSED_MOE 1
  default_var EXL3_MOE_ROW_TILE 0
  default_var EXL3_FAT_KERNEL 1
  default_var EXL3_TEMP_ROWS_FUSED 128
  default_var KV_CACHE_DTYPE fp8_e4m3
  default_var REMOTE_DIR "$HOME/.cache/glm53-profile-$profile"
  default_var CACHE_HOST_PATH "$HOME/.cache/huggingface/vllm-cache-$profile"
  default_var JIT_CACHE_HOST_PATH "$HOME/.cache/glm53-vllm-jit-$profile"
  export SERVE_PROFILE="$profile"
}

show_profile() {
  local names=(SERVE_PROFILE IMAGE MODEL_HOST_PATH DRAFT_HOST_PATH QUANTIZATION MOE_BACKEND VLLM_ATTENTION_BACKEND KV_CACHE_DTYPE USE_CALIBRATED_NVFP4_MLA USE_FP4_INDEXER_CACHE DCP_SIZE GPU_MEMORY_UTILIZATION BLOCK_SIZE MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS ENFORCE_EAGER COMPILATION_CONFIG ENABLE_DFLASH DFLASH_TOKENS DFLASH_DRAFT_TP DFLASH_DRAFT_SAMPLE_METHOD DFLASH_REJECTION_SAMPLE_METHOD ENABLE_DECODE_FIRST_SCHEDULER PREFILL_ADMISSION_POLICY PREFILL_SCHEDULE_INTERVAL LONG_PREFILL_TOKEN_THRESHOLD EXL3_FUSED_MOE EXL3_FAT_KERNEL SYNC_IMAGE_TO_WORKER)
  local name
  for name in "${names[@]}"; do
    printf '%-34s %q\n' "$name" "${!name-}"
  done
}

build_profile() {
  local profile="$1"
  case "$profile" in
    nvfp4-fp4-dcp2)
      "$SCRIPT_DIR/build-production-image.sh"
      "$SCRIPT_DIR/build-fp8-passthrough-image.sh"
      "$SCRIPT_DIR/build-four-over-six-image.sh"
      "$SCRIPT_DIR/build-gscale-tooling-image.sh"
      ;;
    nvfp4-fp8-dcp2|exl3-fp8-dcp2)
      IMAGE="$IMAGE" "$SCRIPT_DIR/build-exl3-fp8-dcp2-image.sh"
      ;;
    exl3-fp4-dcp2)
      IMAGE="$IMAGE" "$SCRIPT_DIR/build-exl3-native-fp4-image.sh"
      ;;
  esac
}

prepare_profile() {
  local profile="$1"
  case "$profile" in
    exl3-*) "$SCRIPT_DIR/prepare-exl3-weights.sh" ;;
    nvfp4-*)
      test -f "$MODEL_HOST_PATH/config.json"
      test -f "$MODEL_HOST_PATH/model.safetensors.index.json"
      test -f "$DRAFT_HOST_PATH/model.safetensors"
      printf 'NVFP4 target and DFlash2 draft are present.\n'
      ;;
  esac
}

action="${1:-list}"
case "$action" in
  list) list_profiles ;;
  stop) exec "$SCRIPT_DIR/stop-cluster.sh" ;;
  show|start|build|prepare)
    [[ $# -eq 2 ]] || { usage >&2; exit 2; }
    configure_profile "$2"
    case "$action" in
      show) show_profile ;;
      start) exec "$SCRIPT_DIR/start-cluster.sh" ;;
      build) build_profile "$2" ;;
      prepare) prepare_profile "$2" ;;
    esac
    ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
