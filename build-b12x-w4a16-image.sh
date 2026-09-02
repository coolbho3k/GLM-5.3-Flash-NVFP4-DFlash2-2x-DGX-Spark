#!/usr/bin/env bash
# Build the opt-in B12X W4A16 A/B image on the head and, by default, dgx1.
# This script never starts or stops a container.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-glm53-v14:nvfp4-gscale-tooling}"
IMAGE="${IMAGE:-glm53-v15:b12x-w4a16-ab}"
BUILD_WORKER="${BUILD_WORKER:-1}"
WORKER_HOST="${WORKER_HOST:-dgx1.lan}"
REMOTE_BUILD_DIR="${REMOTE_BUILD_DIR:-$HOME/.cache/glm53-b12x-w4a16-build}"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)

for command in docker ssh scp; do
  command -v "$command" >/dev/null || {
    echo "Required command not found: $command" >&2
    exit 1
  }
done

docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
  echo "Head is missing base image $BASE_IMAGE" >&2
  exit 1
}

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --file "$SCRIPT_DIR/overlay-dflash2/Dockerfile.b12x-w4a16-ab" \
  --tag "$IMAGE" \
  "$SCRIPT_DIR"

case "$BUILD_WORKER" in
  0) ;;
  1)
    ssh "${ssh_opts[@]}" "$WORKER_HOST" \
      "docker image inspect $(printf '%q' "$BASE_IMAGE") >/dev/null 2>&1"
    ssh "${ssh_opts[@]}" "$WORKER_HOST" \
      "mkdir -p $(printf '%q' "$REMOTE_BUILD_DIR/overlay-dflash2") $(printf '%q' "$REMOTE_BUILD_DIR/b12x_experiment/tests")"
    scp "${ssh_opts[@]}" \
      "$SCRIPT_DIR/overlay-dflash2/Dockerfile.b12x-w4a16-ab" \
      "$WORKER_HOST:$REMOTE_BUILD_DIR/overlay-dflash2/"
    scp "${ssh_opts[@]}" \
      "$SCRIPT_DIR/b12x_experiment/patch_b12x_swiglu_limit.py" \
      "$SCRIPT_DIR/b12x_experiment/patch_b12x_workspace_mode.py" \
      "$SCRIPT_DIR/b12x_experiment/patch_b12x_w4a16_ep.py" \
      "$SCRIPT_DIR/b12x_experiment/verify_b12x_w4a16_install.py" \
      "$WORKER_HOST:$REMOTE_BUILD_DIR/b12x_experiment/"
    scp "${ssh_opts[@]}" \
      "$SCRIPT_DIR/b12x_experiment/tests/test_b12x_swiglu_limit.py" \
      "$SCRIPT_DIR/b12x_experiment/tests/test_b12x_w4a16_ep.py" \
      "$WORKER_HOST:$REMOTE_BUILD_DIR/b12x_experiment/tests/"
    ssh "${ssh_opts[@]}" "$WORKER_HOST" \
      "docker build --build-arg BASE_IMAGE=$(printf '%q' "$BASE_IMAGE") --file $(printf '%q' "$REMOTE_BUILD_DIR/overlay-dflash2/Dockerfile.b12x-w4a16-ab") --tag $(printf '%q' "$IMAGE") $(printf '%q' "$REMOTE_BUILD_DIR")"
    ;;
  *)
    echo "BUILD_WORKER must be 0 or 1" >&2
    exit 2
    ;;
esac

runtime_paths=(
  /usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/b12x_moe.py
  /usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py
  /opt/glm53-tests/test_b12x_w4a16_ep.py
)
local_runtime="$(docker run --rm --entrypoint sha256sum "$IMAGE" "${runtime_paths[@]}")"
if [[ "$BUILD_WORKER" == "1" ]]; then
  printf -v remote_checksum_command '%q ' \
    docker run --rm --entrypoint sha256sum "$IMAGE" "${runtime_paths[@]}"
  remote_runtime="$(ssh "${ssh_opts[@]}" "$WORKER_HOST" "$remote_checksum_command")"
  [[ "$local_runtime" == "$remote_runtime" ]] || {
    echo "B12X runtime files differ between nodes" >&2
    diff <(printf '%s\n' "$local_runtime") <(printf '%s\n' "$remote_runtime") || true
    exit 1
  }
fi

docker image inspect "$IMAGE" --format 'built {{.Id}} as {{join .RepoTags ", "}}'
echo "Image is ready on requested nodes. No server was started or stopped."
