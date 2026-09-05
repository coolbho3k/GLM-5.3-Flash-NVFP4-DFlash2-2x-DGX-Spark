#!/usr/bin/env bash
# Disruptive maintenance window: deliberately leaves serving stopped so the
# caller can build a winning native image before paying for another model load.
# Caller MUST restart the validated recipe or candidate after this script.
set -Eeuo pipefail
RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAT_BUILD="${1:?pass absolute isolated build directory}"
FAT_RESULTS="${2:?pass absolute results directory}"
SANITIZER_ROOT="${SANITIZER_ROOT:-/usr/local/cuda/compute-sanitizer}"
[[ "$FAT_BUILD" = /* && "$FAT_RESULTS" = /* && "$SANITIZER_ROOT" = /* ]]
for variant in control async2 async2_m64; do test -f "$FAT_BUILD/$variant.so"; done
test -x "$SANITIZER_ROOT/compute-sanitizer"
test -d "$FAT_RESULTS"
cd "$RECIPE_ROOT"
./stop-cluster.sh
if docker container inspect vllm_glm53 >/dev/null 2>&1; then
    echo 'Refusing GPU probe: head serving container still exists' >&2
    exit 1
fi
for order in forward reverse; do
    variants=control,async2,async2_m64
    if [[ "$order" = reverse ]]; then variants=async2_m64,async2,control; fi
    docker run --rm --gpus all --network=none --cpus=4 --memory=6g \
        --memory-swap=6g -v "$RECIPE_ROOT/probes:/probes:ro" \
        -v "$FAT_BUILD:/build:ro" --entrypoint timeout \
        glm53-exl3:decode-pipeline-recipe-v1 --kill-after=10 180 \
        python3 /probes/bench_exl3_fat_pipeline.py --libraries /build \
        --variants "$variants" --repeats 30 --rows 129,145,192,256,272,384,512,768,1024,2048,8192 \
        > "$FAT_RESULTS/exl3-fat-tiles-$order.jsonl" \
        2> "$FAT_RESULTS/exl3-fat-tiles-$order.log"
done
for checker in memcheck racecheck synccheck; do
    docker run --rm --gpus all --network=none --cpus=4 --memory=8g \
        --memory-swap=8g -v "$RECIPE_ROOT/probes:/probes:ro" \
        -v "$FAT_BUILD:/build:ro" -v "$SANITIZER_ROOT:/sanitizer:ro" \
        --entrypoint timeout glm53-exl3:decode-pipeline-recipe-v1 --kill-after=10 180 \
        /sanitizer/compute-sanitizer --tool "$checker" --error-exitcode 99 \
        --kernel-name kns=exl3_fat_gemm_kernel \
        python3 /probes/bench_exl3_fat_pipeline.py --libraries /build \
        --variants control,async2,async2_m64 --small-only --repeats 2 \
        > "$FAT_RESULTS/exl3-fat-tiles-$checker.log" 2>&1
done
echo 'Screen finished; serving remains stopped for native-build/restore decision.'
