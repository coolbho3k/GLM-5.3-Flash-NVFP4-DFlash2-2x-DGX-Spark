#!/usr/bin/env bash
# Explicitly disruptive isolated experiment, followed by validated restoration.
set -Eeuo pipefail
RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAT_BUILD="${1:?pass absolute isolated build directory}"
FAT_RESULTS="${2:?pass absolute results directory}"
[[ "$FAT_BUILD" = /* && "$FAT_RESULTS" = /* ]]
test -f "$FAT_BUILD/control.so"
test -f "$FAT_BUILD/async2.so"
test -d "$FAT_RESULTS"
cd "$RECIPE_ROOT"
restore() {
    local screen_rc=$?
    trap - EXIT
    GPU_MEMORY_UTILIZATION=0.87 EXL3_TEMP_ROWS_FUSED=128 \
    EXL3_FUSED_FAT_ACTIVATION=1 EXL3_FAT_ACTIVATION_CONTROL= \
    GLM53_EXL3_MOE_FAST=1 GLM53_EXL3_MOE_STREAM_WEIGHTS=0 \
    TARGET_CUDAGRAPH_SCOPE=c1 IMAGE=glm53-exl3:decode-pipeline-recipe-v1 \
        ./serve-profile.sh start exl3-fp8-dcp2 > "$FAT_RESULTS/fat-pipeline-restore.log" 2>&1
    python3 probes/validate_kv_candidate.py --skip-long \
        > "$FAT_RESULTS/fat-pipeline-restored-functional.jsonl"
    exit "$screen_rc"
}
trap restore EXIT
./stop-cluster.sh
if docker container inspect vllm_glm53 >/dev/null 2>&1; then
    echo 'Refusing GPU probe: head serving container still exists' >&2
    exit 1
fi
for order in forward reverse; do
    variants=control,async2
    if [[ "$order" = reverse ]]; then variants=async2,control; fi
    docker run --rm --gpus all --network=none --cpus=4 --memory=6g \
        --memory-swap=6g -v "$RECIPE_ROOT/probes:/probes:ro" \
        -v "$FAT_BUILD:/build:ro" --entrypoint timeout \
        glm53-exl3:decode-pipeline-recipe-v1 --kill-after=10 180 \
        python3 /probes/bench_exl3_fat_pipeline.py --libraries /build \
        --variants "$variants" --repeats 20 \
        > "$FAT_RESULTS/exl3-fat-pipeline-$order.jsonl" \
        2> "$FAT_RESULTS/exl3-fat-pipeline-$order.log"
done
