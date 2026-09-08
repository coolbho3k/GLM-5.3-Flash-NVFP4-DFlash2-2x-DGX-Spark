#!/usr/bin/env bash
# Apply runtime-selectable source patches before vLLM imports its modules.
set -Eeuo pipefail

python3 /opt/glm53/patch_spinwait.py
python3 /opt/glm53/patch_adaptive_k.py
python3 /opt/glm53/patch_dense_fp8.py
exec vllm serve "$@"
