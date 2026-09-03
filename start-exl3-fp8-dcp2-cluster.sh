#!/usr/bin/env bash
# Compatibility entry point for the unified EXL3 + compact-FP8 profile.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/serve-profile.sh" start exl3-fp8-dcp2
