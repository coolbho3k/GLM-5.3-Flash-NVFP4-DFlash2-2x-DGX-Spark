#!/usr/bin/env bash
# Explicit experiment launcher. Stops both ranks; keeps all other validated knobs.
# Usage: bash probes/restart_draft_sampling.sh greedy|probabilistic
set -Eeuo pipefail
SAMPLING_MODE="${1:?expected greedy or probabilistic}"
case "$SAMPLING_MODE" in
  greedy) SAMPLING_PROPOSAL= ;;
  probabilistic) SAMPLING_PROPOSAL=probabilistic ;;
  *) echo 'expected greedy or probabilistic' >&2; exit 2 ;;
esac
RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RECIPE_ROOT"
export IMAGE=glm53-exl3:prefill-norepack-v1
export GPU_MEMORY_UTILIZATION=0.87 EXL3_TEMP_ROWS_FUSED=128
export EXL3_FUSED_FAT_ACTIVATION=1 EXL3_FAT_ACTIVATION_CONTROL=
export GLM53_EXL3_MOE_FAST=1 GLM53_EXL3_MOE_STREAM_WEIGHTS=0
export EXL3_FAT_PIPELINE=m64_norepack
export EXL3_FAT_PIPELINE_CONTROL=/etc/glm53/adaptive-scheduler/exl3-fat-pipeline.json
export TARGET_CUDAGRAPH_SCOPE=c1 GLM53_SPINWAIT_MS=16
export DFLASH_TOKENS=7 DFLASH_DRAFT_TP=2 DFLASH_DRAFT_VARIANT=mxfp8
export DFLASH_REJECTION_SAMPLE_METHOD=standard
export DFLASH_DRAFT_SAMPLE_METHOD="$SAMPLING_PROPOSAL"
# Resolve/validate the chosen profile before stopping a healthy server.
./serve-profile.sh show exl3-fp8-dcp2
./stop-cluster.sh
if ./serve-profile.sh start exl3-fp8-dcp2; then
  exit 0
fi
if [[ "$SAMPLING_MODE" == probabilistic ]]; then
  echo 'Candidate startup failed; restoring the validated greedy proposal mode.' >&2
  ./stop-cluster.sh
  DFLASH_DRAFT_SAMPLE_METHOD= ./serve-profile.sh start exl3-fp8-dcp2
fi
exit 1
