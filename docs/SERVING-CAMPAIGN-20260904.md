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

For same-boot A/B, the v2 experiment image also accepts
`EXL3_FAT_ACTIVATION_CONTROL=/etc/glm53/adaptive-scheduler/exl3-activation.json`.
This is optional and requires the adaptive scheduler's existing directory
mount. The file contains `{"enabled": false}` or `{"enabled": true}`. Startup
copies it to the worker; when toggling, update both the head file and the
worker's actual deployed file. `serve-profile.sh` uses the profile-specific
directory `~/.cache/glm53-profile-exl3-fp8-dcp2/scheduler_profiles/`; standalone
`start-cluster.sh` instead defaults to `~/.cache/glm53-tp2-deploy/`. Inspect the
container's mount source rather than assuming the deployment directory. Drain
requests before changing it. E2 checks at most once a second; invalid or
missing files retain the last valid setting. Both ranks log actual changes.
The hot file overrides the static activation flag when configured; unset the
control path for ordinary static serving or a static-flag rollback.

The exact integrated helper passed the GPU parity gate with the same error
bounds as the primitive screen. CPU hot-control tests verify static mode,
rate limiting, reversal, and retention of the last valid value on bad input.

Final 512-row screening: C3 mixed decode 7.49 versus 7.59 tok/s/stream and
prefill 222.8 versus 223.8 tok/s. C5 prefill 178.0 versus 178.3; the candidate's
prefill overlapped the full five decoders for only 86% of its lifetime, so its
6.89 versus 6.73 mixed decode is not a clean speedup. C1 included first-use
compilation pauses. Short pure prefill was 1073.5 tok/s, versus about 1089
baseline. There is no convincing scratch-buffer win; restore 128 rows.

The next boot restores GMU 0.87, 128 scratch rows, and starts activation off.
It uses image `glm53-exl3:e2-activation-v2` for an off/on/off same-boot test.
`probes/bench_exl3_tile.py` separately stages an ABI-checked N128/N256 kernel
screen using existing compiled variants. Its beside-server allocation failed
before any timing; no kernel result or serving change exists. Run it only
with the server stopped, regardless of Linux's reported available memory.

The first on passes (`activation-on-b1/b2`) enabled only the head because the
worker copy targeted the standalone launcher's directory, not the profile's
mounted directory. They are explicitly marked invalid as cluster A/B results.
The subsequent both-rank passes verify the actual mounted files before running.

## Fixed-prompt activation result and isolated decode-kernel screens

The stable-prefill probe holds message bytes constant and varies API
`cache_salt`, outside the prompt. The four fixed runs all processed 21,850
tokens with prompt SHA256
`56ab27fb7e244e01f4cdb65e5c7c12666ed8a7c9c426b39ff48437d6fd6e00ec`
and zero global prefix-hit delta. Off a5/a6 averaged 1106.12 input tok/s;
both-rank on b5/b6 averaged 1139.16: **2.99% higher prefill throughput**.
This does not demonstrate a thin-path decode gain.

Both serving containers were then stopped for bounded GPU kernel screens.
Results are synthetic 288-expert, TP-local 4096/1024-dimensional K4 MoE
measurements, not serving tok/s or model-quality evaluations. Probes use fresh
CUDA graphs and independent scratch/locks for each variant. The installed
kernel is timed before and after candidates. Correlated routing shares six
experts within each eight-token speculative block; uniform routing provides
a second distribution. Neither distribution is an actual captured route.

- N128 versus N256: N128 was 11–13% slower; reject.
- Expert groups 8 versus 4 versus 2 blocks: smaller groups sometimes help
  eight-row cases, but regress some correlated larger cases. They also change
  split-K rounding (~0.077% output-relative RMSE in the synthetic screen).
  No global group-size change is justified.
- Shared gate/up input Hadamard: about 1–2% kernel benefit; output differences
  are comparable to stock repeat-run atomic-add rounding (~1e-8 relative).
- Reducing the register pipeline from three stages to one removes compiler-
  reported spills (original: 84 bytes spill stores/188 bytes loads; one stage:
  zero). It improves timings across tested shapes without changing the
  arithmetic result beyond repeat-run noise.
- A 16-wide K tile eliminates spills but is slower; reject. Two shared-memory
  pipeline stages also regress. Four/six stages with a one-stage register
  pipeline perform better than the original three/three pipeline.

`exl3-pipeline-neighbors-isolated.jsonl` uses random nonuniform SUH/SVH scales,
preserving gate/up SUH equality. The 8-block, K32, register-1/shared-6 variant
cuts kernel time roughly 9–12% versus stock across the six tested cases.
This is a candidate, **not yet a serving improvement**. Neighboring pipeline
depths and guarded transform reuse are being screened before native integration.

Reproduce isolated tooling (server stopped, image available):

```bash
mkdir -p /tmp/glm53-exl3-group-probe
docker run --rm --network=none \
  -v "$PWD/probes:/probes:ro" -v /tmp/glm53-exl3-group-probe:/build \
  --entrypoint python3 glm53-exl3:e2-activation-v2 \
  /probes/build_exl3_group_probe.py --output /build --groups 8 \
  --shared-input --frag-stages 1 --shared-stages 6
docker run --rm --gpus all --network=none -e EXL3_TEMP_ROWS_FUSED=128 \
  -v "$PWD/probes:/probes:ro" -v /tmp/glm53-exl3-group-probe:/build:ro \
  --entrypoint timeout glm53-exl3:e2-activation-v2 90 \
  python3 /probes/bench_exl3_groups.py --libraries /build \
  --variants g8_shared_k32_f1_s6 --random-scales
```

These commands create experiment artifacts only. They do not replace the
installed extension, modify checkpoints, or change serving defaults.

### Native opt-in candidate

The final candidate keeps K32/N256, eight thread blocks per expert, one
register stage, and eight shared-memory stages. Twelve shared stages did not
give a consistent additional benefit. `patch_exl3_decode_pipeline.py` adds
two native K4/N256 kernels (shared and independent gate/up SUH) and preserves
the existing stock kernels. A host-side choice selects transform reuse only
when both SUH pointer tables are the identical allocation. The loader aliases
those tables only after its existing all-expert `torch.equal` scale check.

The native dispatcher is gated by `GLM53_EXL3_MOE_FAST=1`, SM121, equal K4
bitrates, and N256-compatible dimensions; other cases keep the original path.
The setting is read at process startup/first use, **not hot-reloadable**.
Changing it requires a restart so captured target graphs match the setting.
An explicitly requested fast path fails startup on an incompatible image;
it must not silently fall back to the Python expert loop.

The compiled native candidate passes graph-replay parity against the original
kernel control (`g8`) for rows 8/48/128, random scale tensors, and both routing
distributions. The separate unequal-SUH stress screen covers rows 1/8/48/128
at 10x input amplitude. Relative output differences remain below 1e-6 (observed
~1e-8), comparable to original-kernel repeat rounding. No weight or KV bytes
are altered. `stock` in these two probe result files is a legacy label for
the installed dispatcher; `native_fast: "1"` identifies it as the candidate.

Both nodes built `glm53-exl3:e2-decode-pipeline-v1`. The image also includes
the previously validated E2 activation helper and a scratch allocation cleanup
(152 MiB/rank of unused reconstruct-path buffers omitted from direct E2).
The optional fresh-build patch applies successfully to the pinned public
EXL3 source as well as the installed source. A full fresh image rebuild is
not yet independently validated during this campaign.

```bash
docker build -f overlay-exl3-fp8/Dockerfile.decode-experiment \
  -t glm53-exl3:e2-decode-pipeline-v1 .
# Build the same image on dgx1, then run A (0) or B (1):
GPU_MEMORY_UTILIZATION=0.87 EXL3_TEMP_ROWS_FUSED=128 \
EXL3_FUSED_FAT_ACTIVATION=1 EXL3_FAT_ACTIVATION_CONTROL= \
GLM53_EXL3_MOE_FAST=0 TARGET_CUDAGRAPH_SCOPE=c1 \
IMAGE=glm53-exl3:e2-decode-pipeline-v1 \
./serve-profile.sh start exl3-fp8-dcp2
```

The serving A/B is pending. Both boots hold the above settings constant except
`GLM53_EXL3_MOE_FAST`. Native off/on must be verified inside both containers.
Do not infer serving throughput from the ~10% isolated kernel-time reduction.

### Serving A control (native pipeline off)

Both ranks verified fast=0, activation=1, scratch=128. At GMU 0.87 the boot
advertised 4,715,025 KV tokens (4.50x the 1,048,576 context limit). Capture
completed normally; no preemptions occurred in the measured requests.
Six 64-token warmups preceded the two 256-token prose/code rounds.

| Concurrency | Prose aggregate tok/s, median | Code aggregate tok/s, median |
| --- | ---: | ---: |
| 1 | 26.75 | 23.30 |
| 3 | 50.65 | 42.62 |
| 6 | 82.45 | 74.88 |

C1 output speed varied with acceptance: prose 27.79/25.70, code 21.25/25.36.
Wall time divided by drafting-cycle count was steadier: 126.18/127.71 ms for
prose and 126.80/129.44 ms for code. This normalization includes first-token
work and is not a GPU timer. Completed-request deltas matched each case's
expected concurrency; there was no evidence of extra completed traffic.

After initial prefill warmup, the cold fixed 21,850-token prompt took 19.255 s
to first token: 1134.75 input tok/s, zero global prefix-cache hit delta, answer
`OK`. The mixed C3 control processed a fresh 5428-token prefill at 218.03 input
tok/s while all three decoders overlapped its full lifetime. Mean per-stream
decode was 10.310 without prefill and 7.976 with prefill; maximum mixed stream
gap was 0.596 s. The synthetic decoder text is not an authoritative benchmark
report; use the measured timestamps/counters only.

A drained successfully. The matching B boot and results follow.

### Serving B candidate (native pipeline on)

Both ranks verified fast=1 with the same image and other settings as A.
Boot advertised 4,743,558 KV tokens (4.52x 1M context); the small difference
from A is boot-profile variance, not a change to KV format or memory fraction.
After six 64-token warmups, two matched 256-token rounds measured:

| Concurrency | Prose A → B aggregate tok/s | Code A → B aggregate tok/s |
| --- | ---: | ---: |
| 1 | 26.75 → 30.13 (+12.7%) | 23.30 → 26.94 (+15.6%) |
| 3 | 50.65 → 54.84 (+8.3%) | 42.62 → 45.73 (+7.3%) |
| 6 | 82.45 → 82.61 (+0.2%) | 74.88 → 81.56 (+8.9%) |

These short runs do not establish those percentages as universal gains.
C1 acceptance increased, while C6 prose acceptance decreased. C1 wall time
per drafting cycle improved more consistently: prose 126.97 → 118.54 ms
(-6.6%), code 127.99 → 120.28 ms (-6.0%). This includes first-token work,
not just GPU execution. Completed-request deltas matched expected traffic.
The native synthetic parity screens remain the numerical evidence; changed
completion hashes or speculative acceptance are not quality measurements.

Warmed cold prefill used the identical 21,850-token prompt SHA with zero
global prefix hits on both boots: 1134.75 → 1138.40 input tok/s (+0.3%),
essentially unchanged. This thin/decode optimization does not target E2
fat-expert prefill kernels.

One matched mixed C3 case improved mean per-stream decode 7.976 → 8.636
tok/s (+8.3%), aggregate decode 23.059 → 24.880 (+7.9%), and concurrent
prefill 218.03 → 235.81 input tok/s (+8.2%). The 5428-token prefill overlapped
all three decoders for its entire lifetime on both runs. Maximum mixed
stream gap changed 0.596 → 0.563 seconds; no preemptions occurred. This is
one synthetic pressure case, not a general mixed-workload guarantee.

Functional checks passed: exact text, arithmetic, bare-prompt coherence,
tool parsing, black-PNG vision, 27,049-token retrieval, and an observed
16,384-token prefix hit. All six concurrent forks also retrieved the correct
answer from a 131,117-token shared prefix, with 737,280 prefix-hit tokens
observed. Warm prefill took 114.575 seconds; fork requests took 14.865–51.330
seconds, so this proves correctness/cache reuse, not low long-context TTFT.
These are regression smoke checks, not a quality
benchmark against the original checkpoint.

Raw B JSONs, warmup, and functional results are in `results/serving-20260904`.
Recompute decode comparisons with:

```bash
python3 probes/compare_serving_quick.py \
  results/serving-20260904/decode-pipeline-A.json \
  results/serving-20260904/decode-pipeline-B.json
```

### Next isolated screen: group-private output

The probe-only `--private-output` builder option reuses the unused up-input
scratch (only with verified shared gate/up SUH) as FP32 output per expert
group. It replaces contended atomic writes with ordinary FP32 additions,
then reduces groups in a separate kernel. Clearing and reduction are timed.
The existing allocation fits up to 64 rows at cap=128, covering C1–C6 K7.
No serving dispatcher/default is changed. This changes FP32 sum order and
must pass the same 1e-6 relative parity gate before any serving consideration.

The probe now records actual active experts, M16 expert tiles, and an
estimated packed-weight throughput. That estimate is not a DRAM hardware
counter: it excludes scales/activations and assumes weight streaming per
expert tile. Do not treat it as proof of bandwidth saturation.
