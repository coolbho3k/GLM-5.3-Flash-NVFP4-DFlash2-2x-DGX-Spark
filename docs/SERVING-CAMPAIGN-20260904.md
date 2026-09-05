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

The serving A/B below holds the above settings constant except
`GLM53_EXL3_MOE_FAST`. Native off/on must be verified inside both containers.
Do not infer serving throughput from the ~10% isolated kernel-time reduction.

The experimental derivative image above is convenient for existing local
images, but is not itself a fresh-clone base. The normal pinned-public-input
builder now includes the additive native patch:

```bash
IMAGE=glm53-exl3:decode-pipeline-recipe-v1 \
  ./serve-profile.sh build exl3-fp8-dcp2
GPU_MEMORY_UTILIZATION=0.87 EXL3_TEMP_ROWS_FUSED=128 \
EXL3_FUSED_FAT_ACTIVATION=1 EXL3_FAT_ACTIVATION_CONTROL= \
GLM53_EXL3_MOE_FAST=1 TARGET_CUDAGRAPH_SCOPE=c1 \
IMAGE=glm53-exl3:decode-pipeline-recipe-v1 \
  ./serve-profile.sh start exl3-fp8-dcp2
```

Startup's existing image-sync path supplies the worker. Model preparation
and host/NIC configuration remain as described in the README. This complete
fresh-build route has not yet been rerun end-to-end during this campaign;
the measured A/B used the derivative image built independently on both nodes.
Fast mode remains opt-in so old installed images and other profiles keep
their existing behavior.

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

Screen outcome (`exl3-private-isolated.jsonl`): **reject**. All ten cases
(rows 1/8/24/48/64, uniform/correlated routing) passed numerical checks,
with relative output RMSE below 8e-8. Private accumulation was generally
1–3% slower than the equivalent shared-input pipeline control; row-1
uniform was about 6% slower. The two row-8 comparisons showed only ~0.5%
benefit against the isolated control, within the native repeat drift.
Neither the serving extension nor its dispatcher includes this variant.

Actual route counts imply about 226–234 GB/s of packed-weight throughput
for most multirow control cases in this screen (one-row cases ~168 GB/s).
Again these are estimates, not measured DRAM utilization. The result makes
output atomic contention a low-priority target at these tested shapes.

Reproduce this screen only with the server stopped:

```bash
docker run --rm --network=none --cpus 2 --memory 2g \
  -v "$PWD/probes:/probes:ro" -v /tmp/glm53-exl3-group-probe:/build \
  --entrypoint python3 glm53-exl3:e2-decode-pipeline-v1 \
  /probes/build_exl3_group_probe.py --output /build --groups 8 \
  --shared-input --frag-stages 1 --shared-stages 8 --private-output
docker run --rm --gpus all --network=none --cpus 4 --memory 8g \
  -e EXL3_TEMP_ROWS_FUSED=128 -e GLM53_EXL3_MOE_FAST=1 \
  -v "$PWD/probes:/probes:ro" -v /tmp/glm53-exl3-group-probe:/build:ro \
  --entrypoint timeout glm53-exl3:e2-decode-pipeline-v1 120 \
  python3 /probes/bench_exl3_groups.py --libraries /build \
  --variants g8_shared_k32_f1_s8_private --rows 1,8,24,48,64 \
  --random-scales --alias-shared-scales
```

The installed native dispatcher is the fast control in this command. To
also include the same isolated control as the saved run, build once without
`--private-output`, then pass both variant labels as in the saved result.
The validated fast=1 server was relaunched after this screen at GMU 0.87.
No further GPU probe should run beside that loaded server.

### Remaining dense-projection lead (read-only attribution)

Grouping the saved C1 worker trace by kernel/grid/graph shows 170 calls to
the BF16 `16x16_128x1` family at grid `(8,99,1)`, totaling 80.18 ms of the
605.68 ms GPU span. The count is consistent with 34 KDA layers over five
steps. Installed `vllm/models/glm5next/nvidia/kda.py` already merges q/k/v/b
and the two low-rank gate inputs into `in_proj_qkvbfg_a`; its TP2 shape is
12576x4096. This shape/count attribution is an inference, not a recorded
module-name trace. It is not six separate launches waiting to be fused.

That matrix contains 103.02 MB of BF16 weights per rank. Dividing by the
observed ~0.472 ms mean kernel time gives ~218 GB/s of weight-only throughput.
As with the EXL3 estimate, this excludes other traffic and does not prove
measured bandwidth saturation. It suggests a targeted BF16 kernel could
yield a modest gain, but does not justify predicting a dramatic speedup.
The generic cuBLAS/Lt/layout screens above already rejected broad changes.

### Restored serving state

After rejecting private output, the validated image was restored with fast=1,
activation=1 and scratch=128 verified inside both ranks. GMU remains 0.87;
this boot advertises 4,757,824 KV tokens (4.54x 1M). The API is healthy.
Two 64-token prose warmup passes covered every concurrency from C1 through
C6. All expected requests completed, with no preemptions or extra completed
requests. First-use pauses at C2/C4/C5 disappeared on the second pass.
These short warmups are not additional A/B performance claims; use the
matched two-round 256-token results above for comparisons.

`serve-profile.sh show` now displays native-fast, activation and scratch
settings explicitly. Defaults remain off/off/128; existing profiles are not
silently switched onto a new extension. Eight CPU recipe-contract tests and
the scoped native-patch, activation, scratch and private-output source tests
passed. Broad host discovery cannot run GPU-dependent tests because host
Python lacks torch; GPU numerical evidence comes from the isolated container
screens and the serving regression gates, not that host discovery command.

The campaign checkpoints are local on `feature/exl3-ab-current`. Public push
was rejected by safety review even after read-only verification of the named
GitHub fork and existing branch. No campaign commits were pushed; main is
unchanged. Publishing requires renewed approval, not a server change.

## Occupancy follow-up: no K7 serving candidate

`exl3-occupancy-isolated.jsonl` records the next bounded screen, with native
fast=1 as installed control. All variants reuse the same synthetic weights,
inputs and routing, and retain K4/N256 quantization. Measured residency:

| Variant | Dynamic shared memory/block | Blocks/SM allowed | Expert groups launched |
| --- | ---: | ---: | ---: |
| K32, register-1/shared-8, tight allocation | 56 KiB | 1 | 6 |
| K16, register-1/shared-8, tight allocation | 36 KiB | 2 | 6 or 12 |
| K16, register-1/shared-4, tight allocation | 26 KiB | 2 | 12 |

The probe checks the mirrored upstream shared-memory layout before compiling,
queries actual CUDA residency, and requires cooperative-launch support for
the expanded 96-block grid. Oversubscribing its persistent cross-block
barriers with a normal launch would be unsafe; unsupported configurations
fail before timing. Source guards have three CPU tests.

The K32 90→56 KiB change keeps output parity at repeat-rounding level, but
changes multirow timing only about -0.24% to +0.15%, within noise. Reject
as a speed optimization. K16 with twice the expert groups reduces one-row
latency ~14–19%, but one row is not our C1 K7 verification workload. At
rows 8/24/48/128 the candidate is generally unchanged or slower; the lone
~2% correlated-row-8 improvement is not a broad win. No serving promotion.

K16 changes floating-point summation order: relative output RMSE against
the K32 control is ~0.00031–0.00034 (0.031–0.034%). It passed the broad
geometry-screen tolerance, **not** the stricter same-math 1e-6 gate. This
is neither checkpoint quantization error nor evidence of better/worse model
quality. There is no reason to accept that extra numerical difference for
the measured K7 workload. No weight, KV or serving kernel change resulted.

Reproduce while serving is stopped, using the existing probe bind mounts:

```bash
python3 /probes/build_exl3_group_probe.py --output /build --groups 8 \
  --shared-input --frag-stages 1 --shared-stages 8 --tight-smem
python3 /probes/build_exl3_group_probe.py --output /build --groups 8 \
  --shared-input --tile-k 16 --frag-stages 1 --shared-stages 8 \
  --tight-smem --resident-blocks 2
python3 /probes/build_exl3_group_probe.py --output /build --groups 8 \
  --shared-input --tile-k 16 --frag-stages 1 --shared-stages 4 \
  --tight-smem --resident-blocks 2
# Run in the GPU-enabled, memory-bounded probe container:
python3 /probes/bench_exl3_groups.py --libraries /build \
  --variants g8_shared_k32_f1_s8_occ1,g8_shared_k16_f1_s8_occ2,g8_shared_k16_f1_s4_occ2 \
  --rows 1,8,24,48,128 --random-scales --alias-shared-scales
```

Build and timing containers must use `GLM53_EXL3_MOE_FAST=1` and scratch=128
as in the preceding screen. Build itself needs no GPU. This concise command
omits the saved screen's K16 single-residency and isolated K32 controls;
their exact compiler commands are in the build manifest.

## Full recipe build verification

The normal `IMAGE=glm53-exl3:decode-pipeline-recipe-v1 ./serve-profile.sh build
exl3-fp8-dcp2` completed successfully from pinned public inputs. Native compile,
version=1, E2 symbols, Python checks and the packaged image contracts passed.
The full wheel build did not reproduce the derivative JIT build's earlier
CUDA-header precedence problem; no header workaround was added to it.

Image identity: `sha256:dc39113a8fce66769da925c2fddcfb58e172b6d73746820a8d5f1f1aebed64cd`.
This closes the image-build gap mentioned in earlier chronological notes;
cluster startup/serving checks on that new image are still in progress.

Cluster startup now includes the installed EXL3 `.so` in the existing
cross-rank runtime checksums. Matching Python overlays alone cannot prove
matching compiled expert kernels. The location is resolved without importing
torch/CUDA and is restricted to the expected installed extension path.
Existing mismatch behavior is unchanged: sync the selected image when
enabled, otherwise fail before launching a mismatched cluster.

### Full-recipe serving verification completed

The image above was streamed through the normal launcher to dgx1 and started
at GMU 0.87, fast=1, activation=1, scratch=128, target graphs C1 and MXFP8
draft graphs C1–C6. Native `.so`, generated kernel source and Python hashes
match between ranks (`decode-pipeline-fresh-{head,worker}.sha256`). The API
is healthy and this image is left serving. Boot KV capacity is 4,800,623
tokens (4.58x 1M); this is ordinary startup variance, not a new KV change.

After an all-concurrency 64-token warmup, the same two-round 256-token
decode tests measured:

| Concurrency | Prose aggregate tok/s | Code aggregate tok/s |
| --- | ---: | ---: |
| 1 | 28.98 | 27.20 |
| 3 | 54.92 | 49.15 |
| 6 | 81.93 | 83.85 |

C1 wall time per drafting cycle was 117.48 ms prose and 117.91 ms code,
versus 126.97/127.99 ms for the original A control. Output throughput still
varies with acceptance; these results reproduce the fast configuration,
not another optimization beyond the previously measured kernel change.
All completed-request counts matched the synthetic workload; no extra
completed traffic or preemptions were observed.

The first long-prefill pass gave 1063.90 input tok/s. Two warmed repeats of
the identical 21,850-token prompt gave 1153.91 and 1153.50 input tok/s
(18.936/18.942 s TTFT), all with zero global prefix-hit delta. Thus the early
sample is not evidence of a persistent regression. CPU binary inspection
also finds identical resource usage for the four relevant serving kernels
(two native fast variants and two E2 fat variants) between derivative and
fresh builds. Raw reports retain build-path-dependent identifiers; only
those known path-dependent namespace prefixes were ignored when matching
the full kernel template/signature suffixes. Resource equality is not a
claim of binary or numerical identity by itself.

The mixed C3 sample gave 8.254 mean per-stream decode tok/s, 23.913 aggregate,
240.16 concurrent prefill input tok/s, and 0.551 s maximum stream gap. This
is consistent with the prior candidate, with the same short-run/acceptance
caveats. All decoders overlapped the full prefill lifetime.

Functional checks passed again: exact response, arithmetic, ordinary prompt,
tools, image input, 27,049-token retrieval and an observed prefix hit. All
six requests against the 131,117-token shared prefix returned MAGNOLIA;
737,280 prefix-hit tokens were recorded. Warm long-prefill time was 114.418 s
versus 114.575 s in the earlier candidate. These are regression checks, not
a comprehensive quality evaluation or a new maximum-context test.

The README now includes a short opt-in build/start command. Existing
defaults remain unchanged. No rejected private-output, K16, tighter-shared-
memory or expanded-grid variant is in this serving image.

## Packed-weight cache-policy screen

The next bounded screen changes only the packed B-matrix async copy helper.
`--weight-cache stream` uses the pinned EXL3 `cp_async_stream` helper (L2
evict-first); `prefetch128` requests 128-byte L2 prefetching. Activation
loads, addresses, copy sizes, predicates, pipeline geometry, arithmetic,
scratch and barriers are unchanged. These are performance hints according
to [NVIDIA's cp.async specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async),
not a different quantization or memory-consistency scheme.

Both variants and the isolated default control compiled with 127 registers
and no spills. SASS inspection confirms the prefetch modifier; streaming
loads use a separate descriptor from activation loads. Merely counting
LDGSTS opcode names misses that descriptor difference.

After stopping both serving containers, two graph-replay passes used the
same synthetic weights, randomized scales, aliased gate/up input transforms,
uniform/correlated routing and rows 8/24/48/128. The second pass reverses
variant order. Each timing is the median of three 40-replay samples; output
clearing is included. `stock` means the installed fast=1 dispatcher here.

| Rows | Stream kernel-time reduction, uniform | Stream kernel-time reduction, correlated |
| --- | ---: | ---: |
| 8 | 0.82% / 1.32% | 2.20% / 1.98% |
| 24 | 2.13% / 0.61% | 1.15% / 1.66% |
| 48 | 2.07% / 1.99% | 4.45% / 2.86% |
| 128 | 3.01% / 3.50% | 3.08% / 3.37% |

Entries are forward/reverse order versus the isolated same-geometry control.
The initial row-8 uniform control drifted about 2% in both passes, so that
specific small gain is inconclusive. Correlated row-8 and larger shapes
make streaming worth a short native serving A/B, not a default promotion.
Prefetch128 shows no consistent gain and is rejected. Worst relative output
RMSE across the screen was 4.735e-8, below the 1e-6 same-math gate. This is
synthetic output parity, not a model-quality evaluation.

Evidence: `exl3-cache-{forward,reverse}.jsonl`, cache build manifest and
SASS load report. The validated full-recipe image is restored automatically
after the screen; no cache-policy experiment was installed into it.

The additive `patch_exl3_stream_weights.py` stages a separate native variant
behind `GLM53_EXL3_MOE_STREAM_WEIGHTS=1` (also requires fast=1). It preserves
the existing fast kernel and generic/E2 GEMM headers byte-for-byte, changes
only a private copy of the decode GEMM helper, and exports its own version
check. Python rejects a requested but unsupported native variant. The
shared and independent-input native kernels compile for SM121 without spills;
CPU source/compatibility guards pass. The experimental Dockerfile is
`overlay-exl3-fp8/Dockerfile.stream-experiment`. The subsequently completed
native serving A/B below rejected promotion: isolated gains did not translate
into a worthwhile serving improvement.

The switch is forwarded to both ranks and shown by the profile launcher;
defaults remain off. The launcher rejects incompatible profiles or a request
without fast/fused MoE. It requires the separate experiment image: do not
enable it against the restored validated image. The normal restore completed
with HTTP 200 and 4,650,826 KV tokens at GMU 0.87. Capacity variation across
these boots is not a cache-format change; no streaming variant is live yet.
Post-restore short functional checks and all four C1/C6 prose/code smoke
cases passed, with the expected completed-request counts and no preemptions.
These 64-token first-use checks are not replacement performance baselines.

Reproduce the isolated screen with the server stopped (CPU-only compilation
may run separately with resource limits):

```bash
python3 /probes/build_exl3_group_probe.py --output /build --groups 8 \
  --shared-input --frag-stages 1 --shared-stages 8 --weight-cache stream
# Also build default and prefetch128 with the same command/options.
GLM53_EXL3_MOE_FAST=1 EXL3_TEMP_ROWS_FUSED=128 \
python3 /probes/bench_exl3_groups.py --libraries /build \
  --variants g8_shared_k32_f1_s8,g8_shared_k32_f1_s8_stream,g8_shared_k32_f1_s8_prefetch128 \
  --rows 8,24,48,128 --repeats 40 --random-scales --alias-shared-scales
```

Use the same GPU-enabled, 8-GiB-limited probe container described above.
Reverse the variant list for the second pass. The sources copied into
`/build` are disposable experiment artifacts, not serving dependencies.

### Native streaming serving A/B

The current-image control was warmed at C1/C3/C6 with both prompts and then
measured with the same two-round, 256-token decode protocol. No unexpected
completed requests were present. `stream-A.json` records C1 prose/code
29.09/27.25 tok/s, C3 aggregate 55.93/50.92, and C6 aggregate 81.75/79.21.
C1 wall time per draft was 118.120/118.944 ms. The identical cold-salted
21,850-token prefill prompt measured 1104.15 input tok/s on first use and
1150.28 on the warmed repeat, both with zero prefix-hit delta.

Only 2.5 GiB of host RAM was available with the loaded server, so both ranks
were stopped before the full native image rebuild. It completed from the
validated base using a minimal four-file build context:

```bash
tar -cf - overlay-exl3-fp8/Dockerfile.stream-experiment \
  overlay-exl3-fp8/rebuild_exl3_extension.py \
  vendor/miaai-exl3/patch_exl3_stream_weights.py vendor/miaai-exl3/exl3.py |
docker build --network=none -f overlay-exl3-fp8/Dockerfile.stream-experiment \
  -t glm53-exl3:decode-stream-v1 -
```

Native build/link and both version-symbol checks passed. Image manifest-list
SHA is `b8ba69e289c0027ceaee79c64f04e92e3f9763773b68dd4693e3a8c50d6b5250`.
The normal launcher transferred the image to dgx1. Both installed native
libraries hash to `7007f20411eb7eef4e6efdb6811e089660716bfe790011d18c6d3c9b8d71911d`.
The generic GEMM, original MoE and validated fast-kernel source hashes match
the control image byte-for-byte. Docker image identifiers were presented
differently by the two daemons; matching installed native binary/content
hashes establish the cross-rank code match for this experiment.

Candidate launch adds only `GLM53_EXL3_MOE_STREAM_WEIGHTS=1` and selects the
new image; GMU 0.87, fast=1, activation=1, scratch=128, C1 target graphs,
MXFP8 drafter graphs C1–C6, K7 and scheduler settings remain unchanged.
Non-secret runtime settings are saved in `stream-{A,B}-runtime.txt` and
`stream-B-worker-runtime.txt`.

Completed results (two rounds, 256 output tokens per request):

| Workload | Control | Streaming | Change |
|---|---:|---:|---:|
| C1 prose tok/s | 29.09 | 29.64 | +1.9% |
| C1 code tok/s | 27.25 | 25.54 | -6.3% |
| C3 aggregate prose tok/s | 55.93 | 53.19 | -4.9% |
| C3 aggregate code tok/s | 50.92 | 50.57 | -0.7% |
| C6 aggregate prose tok/s | 81.75 | 81.63 | -0.2% |
| C6 aggregate code tok/s | 79.21 | 74.45 | -6.0% |
| Warmed uncached prefill input tok/s | 1150.28 | 1132.27 | -1.6% |

C1 wall time per draft changed only 118.120 to 117.517 ms for prose and
118.944 to 118.672 ms for code. Code acceptance differed (32.01% vs 29.16%),
so its throughput regression is not a clean measurement of kernel slowdown.
The conclusion is no worthwhile demonstrated serving gain, not proof that
the cache hint always hurts. Both prefill runs used the identical 21,850-token
prompt with zero prefix hits. Short functionality passed; the candidate was
not taken through mixed-load or long-context qualification after this rejection.

The validated `decode-pipeline-recipe-v1` image was restored with stream=0,
GMU 0.87 and 4,650,826 KV tokens. API health, exact recall, arithmetic, short
coherence, tool calling and image input passed. Evidence is in
`stream-final-restore.log` and `stream-final-functional.jsonl`. The candidate's
5,036,018-token startup budget is boot/profiling variation, not a cache-format
improvement. No weight or KV precision was changed.

### BF16 audit after the streaming experiment

The two dominant BF16 GEMM families total 214.80 ms / 605.68 ms (35.46%)
in the saved five-step C1 worker trace. This is an earlier profiling snapshot,
not a fresh profile of the final restored image. It includes more than one
module family and must not all be labelled target dense projections.

The largest grid group is the inferred KDA fused input described above:
80.18 ms, or 13.24% of the GPU span. Another N=4096 grid group contributes
63.70 ms but mixes several operations; it needs module-level attribution
before a specialized optimization. Two vocabulary-sized groups together
cost 26.52 ms (4.38%); their graph/count/shape signatures are consistent with
target and draft heads, not independently confirmed module annotations.

The EXL3 checkpoint's `ORIGINAL_MODEL_CARD.md` names
`zai-org/GLM-5.3-Flash-BF16`. KDA q/o weights are BF16 even in the earlier
official-FP8-derived repaired checkpoint. A stdlib-only read-only probe,
`probes/audit_exl3_bf16_source.py`, samples 1,152 elements per tensor without
Torch/CUDA or whole-tensor reads. The two sampled KDA weights match exactly.
The sampled dense MLP, shared-expert and MLA BF16 tensors are close to, but
not identical to, the repaired official FP8 tensors dequantized with FP32
block scales and rounded to BF16: 75, 151 and 48 differing elements,
respectively. This disproves exact passthrough in these samples; it is not a
full provenance or model-quality evaluation. Results: `bf16-source-audit.jsonl`.
Consequently, replacing these weights with official FP8 is a precision/source
tradeoff, not an established lossless export repair. It also cannot remove
the native-BF16 KDA bottleneck.

Installed CUDA unquantized dispatch ends in `torch.nn.functional.linear`;
`VLLM_BATCH_INVARIANT` is unset (default false). The existing cuBLAS/Lt probe
therefore tests the relevant default backend, rather than an unrelated path.
For the dominant M=8, N=12576, K=4096 case it measured 456.3 us with cuBLAS,
454.8 us with Lt and 453.7 us on repeated control. Column-major storage did
not help the dominant shape. The `cutlass_80` kernel name alone is not proof
of a broken dispatch on Blackwell.

At 103.02 MB/rank and about 0.472 ms per large KDA input GEMM, the weight-only
estimate is 218 GB/s, versus the Spark's advertised 273 GB/s. This is not a
DRAM-counter measurement, but it makes a 2x quality-neutral speedup of that
operation implausible without changing data reuse. Reaching ideal bandwidth
for that operation alone would save roughly 2.7% of this GPU span. See
[Spark hardware](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) and
[NVIDIA's GEMM memory/compute discussion](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).

Next worthwhile quality-preserving checks are exact-shape/module attribution
of the mixed N=4096 group and a tightly bounded SM121 small-M BF16 kernel
screen with unchanged dtypes and numerical gates. KDA's six input projections
are already fused; a global BLAS or layout switch has already failed. Larger
batch/data reuse and higher speculative acceptance amortize the same BF16
weight reads over more useful tokens, offering a more structural route than
another cache-hint sweep. No new quantization or live kernel change was made
for this audit.

### Upstream-style expert ticket scheduler: isolated backport

The pinned v0.0.43 kernel assigns active experts round-robin to six groups of
eight blocks on GB10. Newer upstream uses greedy tickets instead. Reference:
[`exl3_moe_kernel.cuh` at 499890c75d20d8e7c9d061f37189ae611a5c9f0b](https://github.com/turboderp-org/exllamav3/blob/499890c75d20d8e7c9d061f37189ae611a5c9f0b/exllamav3/exllamav3_ext/quant/exl3_moe_kernel.cuh).
The probe-only `--ticket-scheduler` backports that scheduling mechanism onto
our existing shared-input f1/s8 kernel, preserving GEMM code and the number
of group barriers. It does not import newer quantization formats or change
the installed extension. Scratch size and KV settings remain unchanged.

The isolated harness reserves 1024 ints in its existing lock allocation for
scheduler state, bounds group count to keep it disjoint from barriers, and
checks that next-ticket and retired-group counters return to zero after
graph replays. Last-group retirement uses acquire/release ordering before
reset; scratch is private to each variant and is not concurrently reused.
This probe-only layout is not a production DevCtx integration.

Both kernels compiled with CPU-only containers capped at 1.5 GiB while the
server was live. The control uses 127 registers with no spills; tickets use
128 registers with 16 bytes spill stores and 20 bytes spill loads. Serving
was stopped for GPU checks, then the validated image was restored. The first
screen stopped at a harness shape mismatch in the synthetic empty-routing
case (weights had not been resized to zero routes). Its partial result and
error log are retained. After correcting the harness, the full forward and
reverse variant-order screens both completed.

Each pass tests 8/48/128 rows, uniform/correlated/striped-hot/empty routing,
random scales and shared gate/up SUH, with median-of-three timings over 30
CUDA graph replays. All 24 cases passed finite-output, relative-RMSE <1e-6,
and scheduler-reset gates. These are synthetic checks, not long-context or
serving-quality validation. Empty routing verifies idle-group retirement;
oversized-expert skip behavior is preserved structurally but not separately
GPU-tested in this screen.

| Routing and rows | Ticket kernel time change, forward / reverse |
|---|---:|
| Uniform, 8 | +0.01% / +2.00% |
| Correlated, 8 | +2.95% / +1.68% |
| Uniform, 48 | +1.57% / +1.46% |
| Correlated, 48 | +0.29% / +0.76% |
| Uniform, 128 | +1.99% / +1.91% |
| Correlated, 128 | +0.40% / +0.13% |
| Deliberately striped-hot, 48 | -35.94% / -36.31% |
| Deliberately striped-hot, 128 | -59.68% / -60.00% |

Negative means faster. The intentionally imbalanced case demonstrates the
mechanism but is not representative traffic evidence. Its six hot experts
occupy every sixth active-expert slot, concentrating round-robin work onto
one group. A simplified M16-tile cost model predicts a makespan change from
16 to 9 tiles at 48 rows, and 48 to 17 at 128 rows. For ordinary routing the
same model shows almost no imbalance to fix, matching the lack of timing gain.
It ignores Hadamard, activation, memory contention and scheduling overhead;
it is explanatory, not a GPU performance predictor.

Decision: **do not promote** as a general serving optimization. Real
per-layer route histograms would be needed to justify a conditional ticket
path. C1 has no demonstrated benefit. Candidate registers/spills and atomic
scheduling overhead are possible costs, not independently isolated causes
of the small regressions. No candidate serving A/B was warranted by this screen.

Reproduction tooling:

```bash
# Build each variant with no GPU access in the validated image:
python3 /probes/build_exl3_group_probe.py --output /build --groups 8 \
  --shared-input --frag-stages 1 --shared-stages 8
python3 /probes/build_exl3_group_probe.py --output /build --groups 8 \
  --shared-input --frag-stages 1 --shared-stages 8 --ticket-scheduler
# This wrapper stops serving and restores the validated configuration afterward:
bash probes/run_exl3_ticket_screen.sh /absolute/build /absolute/results
python3 probes/analyze_exl3_ticket_screen.py \
  /absolute/results/exl3-tickets-forward.jsonl \
  /absolute/results/exl3-tickets-reverse.jsonl
```

Use the resource-limited build containers described above; the wrapper bounds
GPU probes to 8 GiB and 180 seconds each. Results, compile manifests and
analysis are in `results/serving-20260904/exl3-tickets-*`. Six CPU ticket-source
and cost-model tests, plus the existing cache-policy source tests, pass.

### Remaining outside-GEMM checks

A closer look at the saved head/worker C1 traces explains the apparent rank
imbalance: one head all-reduce takes 221.234 ms and starts 4.096 ms into the
trace. Removing just that outlier leaves 30.443 ms of head all-reduce time,
versus 29.674 ms total on the worker. The ~220 ms difference is therefore not
evidence of persistently slow networking. Its placement is consistent with a
profiling/start-boundary wait, but the precise cause was not separately traced.

There is a concrete prefill inefficiency in `apply_exl3_batched_fat`: each
fat expert copies immutable gate/up trellises into `packed13` and their output
scales into `svh13` on every invocation. It also stages route counts to the
CPU and loops over fat experts in Python. Thin-expert work overlaps the count
transfer, so the explicit synchronization is not proof of equivalent GPU idle
time. In the old prefill trace, overall GPU idle fraction is only ~1.5%.

The larger-expert `exl3_fat_gemm_kernel` itself loads A/B synchronously and
executes two block barriers for each K16 iteration. It does not use the
asynchronous multistage pipeline added to the small-M decode path. These are
separate quality-preserving opportunities: consume separate gate/up buffers
without runtime repacking; pipeline the large-M GEMM; and eventually move
fat-expert scheduling onto the GPU. The two fat GEMM families total 4.670 s
of the old 18.932 s prefill GPU span. A hypothetical 2x improvement to only
those kernels would imply ~14% overall speedup, not 2x prefill. The saved
trace predates the fused-activation improvement; any new claim needs a fresh
matched measurement. No implementation or current speed-gain claim is made
for these leads yet.

Ticket-screen final restore completed with HTTP 200 and 4,629,427 KV tokens
at GMU 0.87. All short functional checks passed (exact recall, arithmetic,
coherence, tools and image input). The small capacity difference from the
previous 4,650,826-token boot is profiling-budget variation; neither the
ticket variant nor a new weight/KV format is installed. The validated server
is left running. Restore evidence is in `exl3-tickets-restore.log` and
`exl3-tickets-restored-functional.jsonl`.

### Larger-expert prefill pipeline: promising isolated candidate

`build_exl3_fat_pipeline.py` derives a double-buffered variant of the current
E2 kernel. It uses the existing 16-byte `cp.async` helper to prefetch the next
K16 tile into a separate buffer while computing the current tile. M-tail
rows remain explicitly zero-filled without forming invalid global addresses;
the last K iteration issues no out-of-range prefetch. The trellis decoder,
MMA accumulation order, Hadamard output transform, scale application and
scatter code are unchanged. This also changes the load instruction/cache
behavior; speed differences are not attributed solely to overlap.

Both direct and scatter variants compiled for SM121 without spills. Control
register counts are 99/106; async2 counts are 124/126. Shared memory grows from
13,312 to 18,432 bytes per block. This is on-chip scratch, not an additional
persistent model/KV buffer. Runtime occupancy queries accompany each result.
The builder used a CPU-only container limited to 768 MiB while serving was up.

After stopping serving, forward and reverse variant-order screens each ran
14 shapes, including M=1/127/128/129, K=16/32/48 boundary cases, and actual
per-rank gate/up (K4096, N2048) and down (K1024, N4096) dimensions at
129/145/512/2048/8192 rows. Inputs and weights were synthetic; scatter used
unique nontrivial indices and nonzero pre-existing output, including untouched
rows. Every candidate output was **bitwise equal** to the installed native
kernel before and after CUDA-graph replay. All 28 cases passed. Timings are
median-of-three over 20 replays; scatter includes resetting output equally
for every variant. Sanitizer checks have not yet been run.

Representative kernel speedups (control time / async2 time minus one):

| Rows | Projection | Forward / reverse speedup |
|---:|---|---:|
| 129 | Gate/up | +30.2% / +35.2% |
| 145 | Gate/up | +39.4% / +33.5% |
| 512 | Gate/up | -6.9% / +1.7% |
| 2048 | Gate/up | +15.4% / +29.5% |
| 8192 | Gate/up | +15.5% / +17.4% |
| 129 | Down/scatter | +11.9% / +11.0% |
| 512 | Down/scatter | +11.4% / +8.4% |
| 2048 | Down/scatter | +11.3% / +9.6% |
| 8192 | Down/scatter | +7.0% / +9.0% |

The 145-row scatter case showed a first-pass regression but improved on the
reversed pass; it needs confirmation. For 512-row gate/up, comparing against
the installed native kernel rather than the variable isolated control still
shows roughly 6–7% slowdown. Do not enable async2 globally from this screen.
Follow-up row counts are configurable with `--rows`; `--small-only` supports
bounded sanitizer checks. A shape-specific dispatcher may be appropriate,
but is not chosen from these sparse samples.

Fresh control prefill was measured before shutdown. The fixed warmed
21,850-token prompt achieved 1145.92 input tok/s, TTFT 19.068 s, zero prefix
hits (`fat-pipeline-A-prefill-warmed.json`). The preceding two-round file
contains 21,850 and 43,650 input tokens because this harness grows prompts
across rounds; those are not two warmed repeats of an identical prompt.

Decision: retain for native serving validation, **not yet a serving win**.
The additive `patch_exl3_fat_pipeline.py` and
`Dockerfile.fat-pipeline-experiment` stage separate async2 native symbols,
preserving the control CU/CUH and decode implementation. Six CPU structural
tests pass, including control preservation and patch reapplication rejection.
That full native image is not yet built, and no serving dispatcher/default
selects the candidate. The isolated builder and native patch share the same
source transformation. Weight-repacking removal is a separate, unimplemented
lead; this experiment does not include it.

Reproduce the CPU-only build with the validated image and the repository
`probes`/`vendor/miaai-exl3` directories mounted read-only:

```bash
python3 /probes/build_exl3_fat_pipeline.py --output /build \
  --source /recipe-source/exl3_fat_gemm.cu
# Explicitly stops serving and restores the validated image after the screen:
bash probes/run_exl3_fat_pipeline_screen.sh /absolute/build /absolute/results
```

The screen wrapper bounds each GPU process to 6 GiB and 180 seconds. Evidence
is in `exl3-fat-pipeline-{build,forward,reverse}.*`; these probes are not a
replacement for warmed C1/C6, prefill, mixed-load and functional serving gates.

A separate `--extra-tile64` option stages `async2_m64` for follow-up. It reduces
the M tile to 64 and guards the A-copy threads accordingly; the launch grid
and shared-memory calculation derive from the tile size. Gate/up at 129–145
rows currently launches only 32 CTAs (N2048/128 times two M tiles), versus
48 SMs on GB10. An M64 tile launches 48 CTAs for those rows, and compiled
direct/scatter variants use 64 registers with no spills and 14,336 bytes of
shared memory. It may improve parallelism but also rereads weights more often.
This candidate is **CPU-compiled only**, not GPU-qualified, and is not included
in the staged native patch. Its manifests are `exl3-fat-tiles-build.*`; the
original measured DSOs remain separately preserved. No M64 gain is claimed.

Final restoration after the isolated M128 screen completed with HTTP 200,
4,643,693 KV tokens at GMU 0.87, and all short functional checks passing.
The validated image remains live, with no prefill pipeline candidate enabled.
See `fat-pipeline-restore.log` and `fat-pipeline-restored-functional.jsonl`.
