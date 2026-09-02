#!/usr/bin/env bash
# Guarded controller for the Marlin vs FlashInfer B12X W4A16 serving A/B.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-status}"
URL="${URL:-http://127.0.0.1:8000}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_glm53}"
BASELINE_IMAGE="${BASELINE_IMAGE:-glm53-v14:nvfp4-gscale-tooling}"
CANDIDATE_IMAGE="${CANDIDATE_IMAGE:-glm53-v15:b12x-w4a16-ab}"
RESULT_DIR="${RESULT_DIR:-$SCRIPT_DIR/results/b12x-w4a16-ab}"
BASELINE_RESULT="${BASELINE_RESULT:-$RESULT_DIR/marlin.json}"
CANDIDATE_RESULT="${CANDIDATE_RESULT:-$RESULT_DIR/flashinfer-b12x.json}"

require_restart_permission() {
  if [[ "${ALLOW_SERVER_RESTART:-0}" != "1" ]]; then
    echo "Refusing to restart the server." >&2
    echo "Re-run with ALLOW_SERVER_RESTART=1 after explicit authorization." >&2
    exit 2
  fi
}

require_health() {
  curl -fsS --max-time 10 "$URL/health" >/dev/null
}

active_image() {
  docker inspect "$CONTAINER_NAME" --format '{{.Config.Image}}'
}

require_backend() {
  expected="$1"
  docker logs "$CONTAINER_NAME" 2>&1 |
    grep -Fq "Using '$expected' NvFp4 MoE backend" || {
      echo "Active logs do not prove backend $expected." >&2
      exit 1
    }
}

benchmark() {
  label="$1"
  output="$2"
  mkdir -p "$RESULT_DIR"
  require_health
  python3 "$SCRIPT_DIR/probes/bench_prefill_mixed.py" \
    --url "$URL" \
    --label "$label" \
    --output "$output"
}

start_backend() {
  image="$1"
  backend="$2"
  IMAGE="$image" \
    MOE_BACKEND="$backend" \
    ENFORCE_EAGER=1 \
    VLLM_B12X_USE_CUDA_GRAPH=0 \
    GPU_MEMORY_UTILIZATION=0.87 \
    MAX_MODEL_LEN=1048576 \
    DCP_SIZE=2 \
    USE_FP4_INDEXER_CACHE=0 \
    "$SCRIPT_DIR/start-cluster.sh"
}

case "$ACTION" in
  status)
    require_health
    echo "image=$(active_image)"
    docker logs "$CONTAINER_NAME" 2>&1 |
      grep -E "Using '(MARLIN|FLASHINFER_B12X)' NvFp4 MoE backend|GPU KV cache size:" |
      tail -n 3
    ;;
  prepare)
    "$SCRIPT_DIR/build-b12x-w4a16-image.sh"
    ;;
  baseline)
    [[ "$(active_image)" == "$BASELINE_IMAGE" ]] || {
      echo "Baseline server image is not $BASELINE_IMAGE" >&2
      exit 1
    }
    require_backend MARLIN
    benchmark marlin "$BASELINE_RESULT"
    ;;
  start-b12x)
    require_restart_permission
    start_backend "$CANDIDATE_IMAGE" flashinfer_b12x
    require_backend FLASHINFER_B12X
    python3 "$SCRIPT_DIR/probes/validate_kv_candidate.py" \
      --url "$URL" --long-repetitions 7200 --min-long-tokens 30000 \
      --filler-phrase "violet tundra copper harbor "
    ;;
  candidate)
    [[ "$(active_image)" == "$CANDIDATE_IMAGE" ]] || {
      echo "Candidate server image is not $CANDIDATE_IMAGE" >&2
      exit 1
    }
    require_backend FLASHINFER_B12X
    benchmark flashinfer-b12x "$CANDIDATE_RESULT"
    ;;
  compare)
    python3 "$SCRIPT_DIR/probes/compare_prefill_mixed.py" \
      --baseline "$BASELINE_RESULT" \
      --candidate "$CANDIDATE_RESULT"
    ;;
  restore-marlin)
    require_restart_permission
    start_backend "$BASELINE_IMAGE" marlin
    require_backend MARLIN
    ;;
  *)
    echo "Usage: $0 {status|prepare|baseline|start-b12x|candidate|compare|restore-marlin}" >&2
    exit 2
    ;;
esac
