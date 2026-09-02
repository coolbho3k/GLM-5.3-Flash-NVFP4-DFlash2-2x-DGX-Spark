#!/usr/bin/env bash
# Build the opt-in G-scale capture/calibration serving image. Does not run it.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-glm53-v13:nvfp4-four-over-six}"
IMAGE="${IMAGE:-glm53-v14:nvfp4-gscale-tooling}"

command -v docker >/dev/null || {
  echo "Required command not found: docker" >&2
  exit 1
}
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
  echo "Missing base image: $BASE_IMAGE" >&2
  exit 1
}
docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --file "$SCRIPT_DIR/overlay-dflash2/Dockerfile.nvfp4-gscale-tooling" \
  --tag "$IMAGE" \
  "$SCRIPT_DIR"
docker image inspect "$IMAGE" --format 'built {{.Id}} as {{join .RepoTags ", "}}'
