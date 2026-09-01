# EXL3 + native FP8 KV + DCP2

This profile combines MiaAI-Lab's EXL3 routed-expert implementation with the
post-KV-campaign serving stack in this repository:

- EXL3/TR3 4 bpw routed experts and fused EXL3 MoE;
- a native 528-byte GLM NoPE FP8 MLA record;
- FP4 compressed indexer cache;
- TP2 plus target-sequence DCP2;
- exact-fit hybrid allocation;
- replicated KDA and DFlash2 block-table fixes; and
- DFlash2 speculative decoding at K=7 with graph-safe accepted-prefix KDA
  replay.

It lives on branch **feature/exl3-fp8-dcp2**. The original
**feature/reproducible-long-context-serving** profile and launcher are not
modified.

## What is stored

| component | format | distribution |
|---|---|---|
| routed expert weights | EXL3/TR3, nominal 4 bpw | TP2 |
| non-routed target weights | checkpoint native dtype | TP2 |
| target MLA KV | 512 E4M3 bytes + four FP32 group scales = 528 B/token/layer | sequence-sharded DCP2 |
| sparse indexer KV | native FP4 compressed pool | DCP-aware |
| KDA recurrent state | FP16 compact replay | replicated |
| DFlash2 KV/state | auto/bf16, 64-token padded slot-share | TP-local, sequence-replicated |

GLM-5.3 has qk_rope_head_dim=0. The generic packed FP8 MLA ABI reserves
128 bytes per token for a RoPE payload that is always empty for this model.
This profile removes that payload and teaches the existing SparkInfer/B12X
SM120 sparse-MLA reader to consume the resulting 528-byte record.

The older FlashAttention wrapper is not used for this composition. It plans
host-side row lengths before DCP ownership filtering, while the DCP-local
top-k lengths only exist on the GPU. B12X already accepts the runtime
per-row top-k geometry, so extending its existing GLM FP8 layout avoids a
per-layer CPU synchronization and re-plan.

## Pinned inputs

- base image digest: 4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
- SparkInfer: 3a437ab5168060e4d625f05e1625c04089f1ba37
- ExLlamaV3: c5d9c657966ffeeaa9353f0cc899f18629da4a13
- MiaAI-Lab source: 493cb88fc69f8ba73ac87404f429d763e2739d89
- EXL3 checkpoint: Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw at 25a44fdbf16862a46b7cc9921142c6c81350af2f
- DFlash2: incoai/GLM-5.3-Flash-DFlash2 at dc77ff1c99eeb2df044ee3d4f0094eb033fee410

## Build and prepare

On the head:

    ./build-exl3-fp8-dcp2-image.sh
    python3 -m pip install --user -U huggingface_hub
    ./prepare-exl3-weights.sh

The checkpoint preparation script downloads a standalone pinned target tree,
verifies at least 120 safetensors shards, downloads the pinned DFlash2 draft,
and rsyncs both trees to dgx1.lan. Set SYNC_WORKER=0 for a head-only
download. Set MODEL_HOST_PATH, DRAFT_HOST_PATH, or WORKER_HOST when the
defaults do not match a cluster.

The cluster starter streams the local image to the worker only when its tag
is missing, then compares SHA-256 sums of the runtime files on both nodes
before launch.

## Start and stop

Start worker then head and wait for /health:

    ./start-exl3-fp8-dcp2-cluster.sh

Stop both ranks:

    docker rm -f vllm_glm53
    ssh dgx1.lan docker rm -f vllm_glm53

The lower-level launcher is useful for diagnostics:

    # worker first
    ssh dgx1.lan 'cd ~/.cache/glm53-exl3-fp8-dcp2-deploy && ./launch-glm53-exl3-fp8-dcp2.sh 1'
    # head second
    ./launch-glm53-exl3-fp8-dcp2.sh 0

Default serving geometry is the single 1M profile requested for this
campaign:

- max model length: 1,048,576
- maximum sequences: 4
- maximum batched tokens: 2,048
- manager block size: 2,304
- quantization: EXL3
- target KV cache: FP8 E4M3
- decode-context parallel size: 2
- attention backend: FLASHINFER_MLA_SPARSE_SM120
- FP4 indexer cache enabled
- DFlash2 K=7 with draft KV auto
- compact KDA accepted-prefix replay enabled
- CUDA graphs for batch shapes 1, 2, 4, 8, 16, 24, and 32

Do not add **--moe-backend marlin**; that belongs to the NVFP4 target and is
incorrect for EXL3 weights.

Set `COMPACT_SPEC_REPLAY=0` on the cluster starter for the correctness-first
native-snapshot rollback. This allocates all seven recurrent rollback states
and therefore costs KV capacity. `ENFORCE_EAGER=1` remains available for
CUDA-graph diagnostics.

## Validated results on spark-0 + dgx1.lan

The final graph profile was validated on 2026-09-01 with the 1,048,576-token
model limit. UMA state varies between cold boots, so treat the final digits as
observations rather than a fixed ABI.

| measurement | result |
|---|---:|
| logical KV capacity | **2,844,701 tokens** |
| full-1M maximum concurrency | **2.71x** |
| graph memory | 0.98 GiB on the limiting GPU |
| eager logical KV capacity | 2,869,786 tokens |
| fixed K=7 bare-prompt regression | 46 coherent tokens, stop |
| fixed K=7 sustained generation | 512 coherent tokens |
| fixed K=7 concurrent regression | 3/3 coherent, no CUDA error |
| observed concurrent throughput | 22.8 aggregate tok/s |

Ordinary prose remains acceptance-bound, so EXL3 is an experimental
weight-serving alternative rather than the new unconditional production
default. The correctness figures above are from the graph-safe replay build;
older decode numbers from the broken replay lifecycle have intentionally been
removed.

End-to-end correctness gates passed for exact text, deterministic arithmetic,
tool parsing, vision, long-context retrieval, and observable prefix reuse:

| retrieval prompt | exact result | first-request time | follow-up prefix hit |
|---:|---|---:|---:|
| 27,042 tokens | ORCHID | 33.0 s | 10,240 tokens |
| 270,049 tokens | ORCHID | 310.5 s | 256,000 tokens |
| 1,020,049 tokens | ORCHID | 72.4 s with retained cache | 1,003,520 tokens |

The first cold 1.02M attempt stayed compute-active without OOM or preemption
but exceeded the probe's original 900-second HTTP timeout. Its cancelled work
was retained for the successful rerun, so the 72.4-second number is an APC
receipt, not a cold-prefill claim. Use `--request-timeout 1800` for a cold 1M
validation run. Cold million-token prefill is the largest remaining serving
performance weakness.

Three composition bugs were found and fixed during cluster validation:

- DFlash's FlashAttention backend inherited target DCP2 coordinates even
  though its TP-local sliding-window cache is sequence-replicated. Isolating
  it to DCP1 raised deterministic proposal acceptance from 0% to 96% and
  decode from about 7.4 to 55.6 tok/s before graphs.
- CUDA graph replay executes captured tensor copies but does not rerun Python
  assignments. The original compact KDA implementation cleared a Python
  `pending` flag after its first commit and never re-armed it, so recurrent
  state stopped advancing and output eventually repeated. Replay metadata now
  uses fixed persistent buffers updated by every graph execution, and the
  runner commits on every current step that contains draft tokens.
- The `COMPACT_SPEC_REPLAY=0` switch previously disabled replay without
  restoring the seven native rollback-state columns. It now changes both the
  recurrence behavior and the authoritative Mamba cache specification, making
  it a genuine safe fallback.

## Verification gates

The image build fails closed unless EXL3 registration, the fused extension,
the exact 528-byte B12X traits, DCP-aware indexer code, exact-fit allocator,
and both long-context fixes are present. Run the custom writer's numerical
GPU test before a server boot:

    docker run --rm --gpus all --entrypoint python3 \
      glm53-exl3:fp8-dcp2 /opt/glm53/test_fp8_zero_rope_writer.py

The test checks byte-exact E4M3 values, FP32 scales, paged slot addressing,
and negative-slot suppression against a PyTorch reference.

The default uses the validated CUDA-graph shapes with 2K prefill chunks.
`COMPACT_SPEC_REPLAY=0 ./start-exl3-fp8-dcp2-cluster.sh` selects the native
rollback-state fallback.
The exact long-context command used for the final boundary gate was:

    python3 probes/validate_kv_candidate.py \
      --url http://127.0.0.1:8000 \
      --long-repetitions 170000 \
      --min-long-tokens 1000000 \
      --request-timeout 1800
