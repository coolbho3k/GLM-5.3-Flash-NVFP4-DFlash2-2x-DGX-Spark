#!/usr/bin/env bash
# Apply runtime-selectable source patches before vLLM imports its modules.
set -Eeuo pipefail

python3 /opt/glm53/patch_spinwait.py
exec vllm serve "$@"
