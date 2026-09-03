#!/usr/bin/env bash
# Add the EXL3 routed-expert runtime to the calibrated native-FP4 KV image.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-glm53-v14:nvfp4-gscale-tooling}"
IMAGE="${IMAGE:-glm53-exl3:e2-native-fp4-dcp2}"

command -v docker >/dev/null || {
  echo "Required command not found: docker" >&2
  exit 1
}
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
  echo "Missing $BASE_IMAGE; build the NVFP4 production image chain first" >&2
  exit 1
}

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --file "$SCRIPT_DIR/overlay-exl3-native-fp4/Dockerfile" \
  --tag "$IMAGE" \
  "$SCRIPT_DIR"

docker image inspect "$IMAGE" --format 'built {{.Id}} as {{join .RepoTags ", "}}'
