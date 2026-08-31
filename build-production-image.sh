#!/usr/bin/env bash
# Build the validated native-FP4/DCP2/DFlash2 image from pinned public inputs.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPARKINFER_URL="https://github.com/local-inference-lab/sparkinfer.git"
SPARKINFER_COMMIT="3a437ab5168060e4d625f05e1625c04089f1ba37"
SPARKINFER_PATCH="$SCRIPT_DIR/vendor/sparkinfer-glm53-nvfp4.patch"
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6}"
IMAGE="${IMAGE:-glm53-v11:kvopt-final}"
LOCAL_SPARKINFER_DEFAULT="/home/emi/code/glm/vendor/sparkinfer"
BUILD_ROOT="$(mktemp -d -t glm53-kvopt-build.XXXXXXXX)"
SPARKINFER_CONTEXT="$BUILD_ROOT/sparkinfer"
WORKTREE_OWNER=""

cleanup() {
  if [[ -n "$WORKTREE_OWNER" && -e "$SPARKINFER_CONTEXT/.git" ]]; then
    git -C "$WORKTREE_OWNER" worktree remove --force "$SPARKINFER_CONTEXT" \
      >/dev/null 2>&1 || true
  fi
  rm -rf -- "$BUILD_ROOT"
}
trap cleanup EXIT INT TERM

for command in docker git; do
  command -v "$command" >/dev/null || {
    echo "Required command not found: $command" >&2
    exit 1
  }
done
test -s "$SPARKINFER_PATCH"

if [[ -n "${SPARKINFER_REPO:-}" ]]; then
  source_repo="$SPARKINFER_REPO"
elif [[ -d "$LOCAL_SPARKINFER_DEFAULT/.git" ]]; then
  source_repo="$LOCAL_SPARKINFER_DEFAULT"
else
  source_repo=""
fi

if [[ -n "$source_repo" ]]; then
  git -C "$source_repo" cat-file -e "$SPARKINFER_COMMIT^{commit}"
  git -C "$source_repo" worktree add --detach \
    "$SPARKINFER_CONTEXT" "$SPARKINFER_COMMIT"
  WORKTREE_OWNER="$source_repo"
else
  git clone --filter=blob:none "$SPARKINFER_URL" "$SPARKINFER_CONTEXT"
  git -C "$SPARKINFER_CONTEXT" checkout --detach "$SPARKINFER_COMMIT"
fi

git -C "$SPARKINFER_CONTEXT" apply --check "$SPARKINFER_PATCH"
git -C "$SPARKINFER_CONTEXT" apply "$SPARKINFER_PATCH"

docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull "$BASE_IMAGE"
docker build \
  --build-context "sparkinfer=$SPARKINFER_CONTEXT" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --file "$SCRIPT_DIR/overlay-dflash2/Dockerfile.reproducible" \
  --tag "$IMAGE" \
  "$SCRIPT_DIR"

docker image inspect "$IMAGE" --format 'built {{.Id}} as {{join .RepoTags ", "}}'
