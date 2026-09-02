#!/usr/bin/env bash
# Add the GLM mixed NVFP4/W8A16 loader fix to a known-good serving image.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-glm53-v11:kvopt-final}"
IMAGE="${IMAGE:-glm53-v12:fp8-passthrough}"

for command in docker; do
  command -v "$command" >/dev/null || {
    echo "Required command not found: $command" >&2
    exit 1
  }
done

docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
  echo "Missing base image: $BASE_IMAGE" >&2
  exit 1
}

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --file "$SCRIPT_DIR/overlay-dflash2/Dockerfile.fp8-passthrough" \
  --tag "$IMAGE" \
  "$SCRIPT_DIR"

docker image inspect "$IMAGE" --format 'built {{.Id}} as {{join .RepoTags ", "}}'
