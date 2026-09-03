# EXL3 + native FP8 KV + DCP2

This profile combines MiaAI-Lab's EXL3 routed-expert implementation with the
post-KV-campaign serving stack in this repository:

- EXL3/TR3 4 bpw routed experts and fused EXL3 MoE;
- a native 528-byte GLM NoPE FP8 MLA record;
- FP8 E4M3 indexer cache;
- TP2 plus target-sequence DCP2;
- exact-fit hybrid allocation;
- replicated KDA and DFlash2 block-table fixes; and
- DFlash2 speculative decoding at K=7 with graph-safe accepted-prefix KDA
  replay.

The unified selectable recipe lives on branch **feature/exl3-ab-current**.
It retains the post-KV-campaign NVFP4 profile and adds EXL3 with either target-KV format.

## What is stored

| component | format | distribution |
|---|---|---|
| routed expert weights | EXL3/TR3, nominal 4 bpw | TP2 |
| non-routed target weights | checkpoint native dtype | TP2 |
| target MLA KV | 512 E4M3 bytes + four FP32 group scales = 528 B/token/layer | sequence-sharded DCP2 |
| sparse indexer KV | FP8 E4M3 | DCP-aware |
| KDA recurrent state | FP16 compact replay | replicated |
| DFlash2 KV/state | auto/bf16, exact-fit slot-share | TP-local, sequence-replicated |

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
- MiaAI-Lab source: eb0469fbb2b49fd7c025f594a3339a121e58f7a9
- EXL3 checkpoint: Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw at 25a44fdbf16862a46b7cc9921142c6c81350af2f
- DFlash2: incoai/GLM-5.3-Flash-DFlash2 at dc77ff1c99eeb2df044ee3d4f0094eb033fee410

## Build and prepare

From a fresh clone on the head, install the Hugging Face CLI, download the
pinned public EXL3 and DFlash2 checkpoints, build the selected image, and
start the cluster:

    python3 -m pip install --user -U 'huggingface_hub[cli]'
    ./serve-profile.sh prepare exl3-fp8-dcp2
    ./serve-profile.sh build exl3-fp8-dcp2
    ./serve-profile.sh start exl3-fp8-dcp2

The checkpoint preparation script downloads a standalone pinned target tree,
verifies at least 120 safetensors shards, downloads the pinned DFlash2 draft,
and rsyncs both trees to `dgx1.lan`. Set `SYNC_WORKER=0` for a head-only
download. Set `MODEL_HOST_PATH`, `DRAFT_HOST_PATH`, or `WORKER_HOST`
when the defaults do not match a cluster.

The cluster starter streams the local image to the worker only when its tag
is missing, then compares SHA-256 sums of the runtime files on both nodes
before launch.

The selector exposes all four weight/KV compositions:

    ./serve-profile.sh list
    ./serve-profile.sh show exl3-fp8-dcp2

The native-FP4 EXL3 image layers on the NVFP4 production image chain. On a
brand-new Docker cache, build `nvfp4-fp4-dcp2` before
`exl3-fp4-dcp2`. The compact-FP8 EXL3 image is self-contained and is the
recommended path for reproducing the current live result.

## Start and stop

The selector starts the worker, then the head, and waits for `/health`:

    ./serve-profile.sh start exl3-fp8-dcp2

Stop both ranks:

    ./serve-profile.sh stop

The compatibility wrapper `./start-exl3-fp8-dcp2-cluster.sh` selects the
same profile. The lower-level launcher remains available for diagnostics.

The validated EXL3+FP8 defaults are:

- max model length: 1,048,576
- maximum sequences: 6
- maximum batched tokens: 8,192
- E2 fat-expert prefill kernel enabled
- E2 scratch sized against the 8,192-token scheduler budget
- stock SpinCondition reader window
- CLI block size: 2,048 (attention-manager block size: 4,096)
- quantization: EXL3
- target KV cache: FP8 E4M3
- decode-context parallel size: 2
- attention backend: FLASHINFER_MLA_SPARSE_SM120
- FP8 E4M3 indexer cache
- DFlash2 K=7 with draft KV auto
- compact KDA accepted-prefix replay enabled
- decode-first adaptive scheduler enabled
- eager execution

Do not add **--moe-backend marlin**; that belongs to the NVFP4 target and is
incorrect for EXL3 weights.

Set `EXL3_FAT_KERNEL=0` to select the pre-E2 expert path without rebuilding.
Set `GLM53_SPINWAIT_MS=stock` to restore vLLM's one-second reader spin
window after overriding it. The 8,192-token budget is the current mixed-load
default and can be reduced with `MAX_NUM_BATCHED_TOKENS`.
The upstream indexer-workspace policy is also not stacked over this repo's
existing long-context workspace bound; changing that allocator remains a
separate concurrency-versus-capacity experiment.

Set `COMPACT_SPEC_REPLAY=0` on the cluster starter for the correctness-first
native-snapshot rollback. This allocates all seven recurrent rollback states
and therefore costs KV capacity. `ENFORCE_EAGER=1` explicitly selects eager
execution.

The 2,048 CLI block size is an exact-fit fix, not just a tuning knob. It makes
the target FP8 page 528 x 4,096 bytes and the paired draft block 2,112 tokens,
which is divisible by 32. The old 2,304 setting yielded a 2,640-token draft
block and forced a 64-token padded fallback, wasting roughly 41 times the
intended draft-cache allocation.

Piecewise CUDA graphs remain opt-in for diagnostics:

    ENFORCE_EAGER=0 \
    COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"PIECEWISE"}' \
      ./serve-profile.sh start exl3-fp8-dcp2

On this stack, that mode captured 15 graphs using 0.21 GiB but reduced both
decode and prefill throughput, so it is deliberately not the default.

## Validated results on spark-0 + dgx1.lan

The final eager EXL3+FP8 profile was validated on 2026-09-03 at a
1,048,576-token model limit. UMA state varies between boots, so capacity is
reported as an observed range rather than a fixed ABI.

| measurement | result |
|---|---:|
| logical KV capacity | **4.59M to 4.73M tokens** |
| full-1M maximum concurrency | **4.39x to 4.51x** |
| C1 decode (256 output tokens) | **30.5 tok/s** |
| C6 aggregate decode | **64.4 tok/s** |
| C6 per-request decode | **13.2 tok/s** |
| 5,472-token pure prefill | **about 1,072 tok/s** |
| 87,352-token pure prefill | **about 1,091 tok/s** |
| 34,135-token prefill under decode load | **592 to 636 tok/s** |
| decode retained during mixed load | **44% to 48%** |

The EXL3+native-FP4 target-KV profile also passed correctness at 1M and
observed 7,797,743 logical KV tokens. It reached 31.7 tok/s at C1 and
48.3 aggregate tok/s at C6; lower speculative acceptance made FP8 the better
parallel-decode balance in this run.

The opt-in piecewise-graph A/B reached 27.3 tok/s at C1 and 57.8 aggregate
tok/s at C6, about 10% below the fresh eager baseline. Its 43,656-token
prefill was about 1,016 tok/s versus roughly 1,095 tok/s eager.

Ordinary prose remains acceptance-bound, so EXL3 is an experimental
weight-serving alternative rather than the new unconditional production
default. The correctness figures above use graph-safe replay code in eager
mode; older decode numbers from the broken replay lifecycle have intentionally
been removed.

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
and both long-context fixes are present. Host-only recipe checks are:

    bash -n serve-profile.sh start-cluster.sh \
      launch-glm53-vllm-tp2-dflash2.sh
    python3 probes/test_serving_profiles.py
    python3 probes/test_upstream_e2_recipe.py
    python3 probes/test_chat_template_prefix.py

With the server stopped and both GPUs available, run the numerical kernel
gates before a new image is served:

    docker run --rm --gpus all --entrypoint python3 \
      glm53-exl3:e2-fp8-dcp2 /opt/glm53/test_fp8_zero_rope_writer.py
    docker run --rm --gpus all --entrypoint python3 \
      glm53-exl3:e2-fp8-dcp2 /opt/glm53/test_exl3_e2_kernel.py

The tests check byte-exact E4M3 values, FP32 scales, paged slot addressing,
negative-slot suppression, and the additive E2 CUDA kernel against reference
implementations.

`COMPACT_SPEC_REPLAY=0 ./serve-profile.sh start exl3-fp8-dcp2` selects the
native rollback-state fallback. The exact long-context boundary gate is:

    python3 probes/validate_kv_candidate.py \
      --url http://127.0.0.1:8000 \
      --long-repetitions 170000 \
      --min-long-tokens 1000000 \
      --request-timeout 1800
