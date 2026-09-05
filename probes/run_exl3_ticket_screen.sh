#!/usr/bin/env bash
# Stop serving, run bounded isolated kernels, then restore the validated image.
# Usage: bash probes/run_exl3_ticket_screen.sh /absolute/build /absolute/results
set -Eeuo pipefail
RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TICKET_BUILD="${1:?pass the isolated build directory}"
TICKET_RESULTS="${2:?pass the result directory}"
[[ "$TICKET_BUILD" = /* && "$TICKET_RESULTS" = /* ]]
test -f "$TICKET_BUILD/g8_shared_k32_f1_s8.so"
test -f "$TICKET_BUILD/g8_shared_k32_f1_s8_tickets.so"
test -d "$TICKET_RESULTS"
cd "$RECIPE_ROOT"

restore() {
    local screen_rc=$?
    trap - EXIT
    GPU_MEMORY_UTILIZATION=0.87 EXL3_TEMP_ROWS_FUSED=128 \
    EXL3_FUSED_FAT_ACTIVATION=1 EXL3_FAT_ACTIVATION_CONTROL= \
    GLM53_EXL3_MOE_FAST=1 GLM53_EXL3_MOE_STREAM_WEIGHTS=0 \
    TARGET_CUDAGRAPH_SCOPE=c1 IMAGE=glm53-exl3:decode-pipeline-recipe-v1 \
        ./serve-profile.sh start exl3-fp8-dcp2 > "$TICKET_RESULTS/exl3-tickets-restore.log" 2>&1
    python3 probes/validate_kv_candidate.py --skip-long \
        > "$TICKET_RESULTS/exl3-tickets-restored-functional.jsonl"
    exit "$screen_rc"
}
trap restore EXIT
./stop-cluster.sh
if docker container inspect vllm_glm53 >/dev/null 2>&1; then
    echo 'Refusing GPU probe: head serving container still exists' >&2
    exit 1
fi

for order in forward reverse; do
    variants=g8_shared_k32_f1_s8,g8_shared_k32_f1_s8_tickets
    if [[ "$order" = reverse ]]; then
        variants=g8_shared_k32_f1_s8_tickets,g8_shared_k32_f1_s8
    fi
    docker run --rm --gpus all --network=none --cpus=4 --memory=8g \
        --memory-swap=8g -e EXL3_TEMP_ROWS_FUSED=128 -e GLM53_EXL3_MOE_FAST=1 \
        -v "$RECIPE_ROOT/probes:/probes:ro" -v "$TICKET_BUILD:/build:ro" \
        --entrypoint timeout glm53-exl3:decode-pipeline-recipe-v1 --kill-after=10 180 \
        python3 /probes/bench_exl3_groups.py --libraries /build \
        --variants "$variants" --rows 8,48,128 \
        --routing uniform,correlated,striped_hot,empty --repeats 30 \
        --random-scales --alias-shared-scales \
        > "$TICKET_RESULTS/exl3-tickets-$order.jsonl" \
        2> "$TICKET_RESULTS/exl3-tickets-$order.log"
done
