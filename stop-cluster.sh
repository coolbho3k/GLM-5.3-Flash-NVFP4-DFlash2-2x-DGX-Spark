#!/usr/bin/env bash
set -Eeuo pipefail

WORKER_HOST="${WORKER_HOST:-dgx1.lan}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_glm53}"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" \
  "docker rm -f $(printf '%q' "$CONTAINER_NAME") >/dev/null 2>&1 || true"
echo "Stopped $CONTAINER_NAME on both nodes."
