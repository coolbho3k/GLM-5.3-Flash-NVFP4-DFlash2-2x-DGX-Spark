#!/usr/bin/env bash
# Build the ABI-compatible NVFP4 four-over-six KV-writer candidate.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-glm53-v12:fp8-passthrough}"
IMAGE="${IMAGE:-glm53-v13:nvfp4-four-over-six}"

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
  --file "$SCRIPT_DIR/overlay-dflash2/Dockerfile.nvfp4-four-over-six" \
  --tag "$IMAGE" \
  "$SCRIPT_DIR"

docker image inspect "$IMAGE" --format 'built {{.Id}} as {{join .RepoTags ", "}}'
