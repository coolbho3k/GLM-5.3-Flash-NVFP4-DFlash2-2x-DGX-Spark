# Serving optimization campaign, 2026-09-04

Objective: improve end-to-end serving on two Sparks, prioritizing C1-C6
decode and mixed chat/agent workloads, without further target quantization.
Preserve KV capacity where possible; measure any workspace/graph trade.

## Baseline

Git base: cb5d8c4. Live image: glm53-exl3:e2-fp8-dcp2-target-cg-v1.
EXL3 target, compact FP8 MLA and FP8 indexer, MXFP8 DFlash2 TP2 K7,
TP2/DCP2, 1M context, GMU 0.87, six sequences, MNBT 8192, E2 enabled,
128 fused expert scratch rows, 16 ms spinwait, draft graphs C1-C6.
The live server has selective FULL_DECODE_ONLY target capture at 8 tokens
(C1), although the public profile defaults to eager target execution.

Adaptive policy interactive-adaptive-v3: 256/2 for C1-C2, 256/4 for
C3-C4 (up to 384 with competing prefills), 256/6 for C5-C6.

## Sequence and gates

1. Measure current C1/C3/C5 mixed pressure and C1/C6 decode; capture short
   profiles to attribute routing, expert, attention, draft, and communication.
2. Compare smaller mixed packets against the baseline on the same boot.
   Test the EXL3 128-total-row boundary, accounting for speculative rows.
3. Compare pure-prefill budgets 2048/4096/8192, initially through the existing
   hot-reloadable pure-prefill threshold; verify a winning launch-time budget
   separately because scratch reservation and multiple prefills differ.
4. Use profile evidence to target routing fusion/GPU-driven expert dispatch,
   concurrency-dependent speculation, or selective target graph expansion.
5. Validate winners on varied decode prompts, cold prefill, mixed pressure,
   correctness, prefix reuse, and long context. Record actual startup KV.
6. Promote only reproducible improvements, document rollback, and leave the
   winning server healthy. Keep rejected experiments identifiable.

Results live in results/serving-20260904/. Use unique prefill salts, stable
decoder prompts, warmed paths, alternating controls, and reported actual
token counts. Record cached prompt tokens and the fraction of prefill spent
at the intended decode concurrency. A full-wave throughput number is not a
measurement of contention alone. Sampling/acceptance changes must be
distinguished from changes in engine-step cost.

The first exploratory baseline used the older harness. Subsequent runs add
exact stream finish timestamps, output hashes, cache-hit receipts, and
prefill overlap. Compare per-stream rates with that first run; its aggregate
decoder wall time may include time spent waiting for prefill to finish.

## Bounded profiling

The launcher accepts optional `PROFILER_CONFIG` JSON and forwards it to both
ranks. It is unset for ordinary serving. A diagnostic boot can use:

```bash
GPU_MEMORY_UTILIZATION=0.80 TARGET_CUDAGRAPH_SCOPE=c1 \
PROFILER_CONFIG='{"profiler":"torch","torch_profiler_dir":"/cache/serving-profiles","torch_profiler_with_stack":false,"torch_profiler_record_shapes":false,"torch_profiler_with_memory":false,"ignore_frontend":true,"delay_iterations":2,"max_iterations":5}' \
  ./serve-profile.sh start exl3-fp8-dcp2
```

Start/stop the profiler through the API around a warmed short workload.
Traces are in each rank's profile-specific cache under `serving-profiles`.
Profiled throughput is diagnostic only; compare speed with capture inactive.

## First mixed-packet screening

Same live boot; 384 output tokens per decoder, 10,885-token fresh prefill.
Prompts used unique prefix salts; this server omits the cache-hit usage field,
so zero cache hits are not independently confirmed. The entire 64/1 prefill
ran under the intended decoder count. It is rejected as a throughput policy.

| Decoders | Baseline mixed tok/s/stream | 64/1 mixed tok/s/stream | Baseline prefill tok/s | 64/1 prefill tok/s | Baseline p95 gap | 64/1 p95 gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.00 | 8.09 | 424.8 | 234.0 | 0.494 s | 0.307 s |
| 3 | 7.59 | 6.43 | 223.8 | 189.5 | 0.541 s | 0.359 s |
| 5 | 6.73 | 5.59 | 178.3 | 165.5 | 0.576 s | 0.400 s |

Control decode at C3/C5 was stable across both runs (10.31/10.28 and
8.432/8.431 tok/s/stream). C1 acceptance varied more. Smaller packets reduce
individual pauses but incur mixed-step overhead too frequently. Avoiding the
EXL3 host-count path alone did not make 64/1 worthwhile. Baseline restored.

Single cold synthetic-prefill screening (21,853-21,854 actual input tokens):
8192 cap 1088.9 input tok/s, 2048 cap 1052.1, 4096 cap 1094.1. These are
hot-reloaded per-request caps with the original 8192 scratch reservation,
not different launch-time MNBT settings. No convincing throughput winner;
do not promote a change from these single samples.

## Profile evidence and next screens

Five-step traces are diagnostic, not serving throughput measurements. The C1
worker trace has a 605.7 ms GPU span: 259.5 ms in EXL3 expert kernels and
214.8 ms in two BF16 GEMM families. The head includes additional collective
waiting at profiler startup; do not attribute that difference to network
bandwidth. C6 head EXL3 kernels total 549.7 ms over a 1140.1 ms span.
The mixed trace includes a large host gap during capture/JIT activity; its
gap fraction is not evidence of ordinary production GPU underutilization.

Pure-prefill trace: EXL3 E2 gate/up and down GEMMs total 4.67 seconds over an
18.93-second GPU span. Activation clamp/multiply/copy and related E2 staging
are visible costs. The traced 87,389-token request's 106-second TTFT includes
profiling overhead and is not comparable to the ordinary 1090 tok/s baseline.

`probes/bench_bf16_blas.py` compares CUDA-graph replay of native BF16 linear
operations, identical inputs, cuBLAS/cuBLASLt, and a repeated control. The
backend-only screen found no broad improvement; a column-major weight layout
also regressed the dominant 12576x4096 C1 shape. Some small shapes improve,
but this does not justify a global backend/layout change. Some Lt choices
also changed accumulation error despite unchanged input/output dtypes. Neither
change has been applied to serving. These are synthetic kernel screens, not
model accuracy evaluations. Retained results include FP32-reference errors.

The first probe allocation failed beside the post-profiling server; health
remained 200. Both ranks were then stopped for the isolated BF16 screens.
The next boot uses GMU 0.80, 512 fused scratch rows, and no profiler. All
other serving settings remain baseline. The four shared EXL3 scratch buffers
increase from about 15 to 60 MiB per rank; the actual startup KV pool is the
authority for capacity. Test mixed throughput, C1/C6, and pure prefill before
promoting this setting: larger scratch also changes which experts use E2.

512-row diagnostic boot: 1,925,955 KV tokens versus 2,018,687 on the prior
128-row diagnostic boot. This is the observed whole-boot difference, not an
attribution solely to the additional 45 MiB of fused scratch. Both ranks
became healthy. The launcher observer exited with a parse error after its
script was edited during execution; `bash -n` passes on the resulting file.
Do not edit an executing shell script during subsequent boots.

## Opt-in E2 activation fusion (not yet serving-validated)

`EXL3_FUSED_FAT_ACTIVATION=1` replaces E2's separate FP32 clamp, sigmoid,
multiply operations with the existing vLLM `silu_and_mul_with_clamp` CUDA op
(alpha=1, beta=0). It retains the FP32 intermediate and the original FP16 cast
before the down-projection Hadamard transform. No weight/cache formats change,
and thin-expert decode does not enter this path. Default remains 0 for A/B.

The primitive test at 145/512/2048/8192 rows showed about 2.5-4x faster
activation processing. This is not a whole-model speedup: the trace suggests
only a few percent prefill opportunity. Maximum FP32 difference was 1.53e-5;
roughly 0.0007-0.0014% of FP16 outputs rounded differently in the synthetic
test. This is floating-point reassociation, not new quantization. The exact
installed helper has a separate `--integration` parity gate.

Fast overlay for machines with the current image already built:

```bash
docker build -f overlay-exl3-fp8/Dockerfile.activation-experiment \
  -t glm53-exl3:e2-activation-v1 .
# Run GPU parity only with sufficient headroom and no serving benchmark active:
docker run --rm --gpus all --entrypoint python3 glm53-exl3:e2-activation-v1 \
  /opt/glm53/bench_exl3_fat_activation.py --integration
EXL3_FUSED_FAT_ACTIVATION=1 IMAGE=glm53-exl3:e2-activation-v1 \
  TARGET_CUDAGRAPH_SCOPE=c1 ./serve-profile.sh start exl3-fp8-dcp2
```

Build the overlay on both ranks, or use the launcher's image synchronization.
The full EXL3 recipe also copies the updated vendor implementation; the
overlay is only a fast local experimentation path, not a new prerequisite.
Rollback is `EXL3_FUSED_FAT_ACTIVATION=0` or the previous image.
