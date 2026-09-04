# GLM-5.3-Flash on 2× DGX Spark

Serve [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
across two NVIDIA DGX Spark systems with an OpenAI-compatible vLLM API. This
repository contains the patched runtime, pinned image builds, model download
helpers, and a two-node launcher.

> **Canonical recommended setup:** `exl3-fp8-dcp2` — EXL3 target weights,
> FP8 target KV cache, an MXFP8 DFlash2 drafter, TP2/DCP2, and the adaptive
> decode-first scheduler. This is the best overall balance here for model
> quality, decode speed, mixed chat/agent workloads, and long-context capacity.

The recommended profile has been exercised at a 1,048,576-token model limit,
with one through six concurrent decode sessions and million-token retrieval.
On our two-Spark cluster it exposes more than 4.5 million logical KV tokens;
the exact pool varies with unified-memory state at startup.

## What this repository adds

- A fused EXL3 MoE path for GLM-5.3, including a larger prefill-oriented expert
  kernel on ARM64/SM121.
- A compact 528-byte FP8 NoPE MLA cache and FP8 sparse-indexer cache for GB10.
- DCP2 sequence sharding for the target cache while keeping DFlash2 state
  correctly replicated.
- Native B12X MXFP8 drafter GEMMs and drafter-only CUDA graphs for concurrency
  levels C1 through C6.
- Graph-safe compact replay of GLM's recurrent KDA state.
- An adaptive `AsyncScheduler` policy that protects interactive decode while
  continuing to admit prefill under load.
- Long-context allocation, block-table, indexer-workspace, and SM121 kernel
  fixes needed to serve this model reliably.
- One-command two-node startup, cross-rank runtime verification, model/image
  synchronization, health checking, and cleanup after a failed launch.
- OpenAI-compatible chat, tool calling, vision, and common reasoning-effort
  aliases.

The lower-level fixes and measurements are documented under [`docs/`](docs/).
The main path below is intentionally short.

## Recommended configuration

| Component | Recommended setting |
|---|---|
| Profile | `exl3-fp8-dcp2` |
| Target weights | [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) |
| Target KV | FP8 E4M3, compact NoPE MLA layout |
| Sparse indexer | FP8 E4M3 |
| Parallelism | TP2 across two nodes, DCP2 for target decode |
| Speculation | DFlash2, K=7, draft TP2 |
| Drafter | [MXFP8 DFlash2](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8) |
| CUDA graphs | Drafter C1–C6; target eager |
| Context limit | 1,048,576 tokens |
| Concurrent sequences | Up to 6 by default |
| Scheduler | Adaptive decode-first `AsyncScheduler` |
| GPU memory utilization | 0.87 |

The launcher selects all of these defaults when given
`exl3-fp8-dcp2`; they do not need to be entered separately.

## KV cache: what changed and what to expect

GLM-5.3 uses NoPE MLA, so its cache does not need the 128-byte RoPE field
reserved by the generic packed FP8 layout. The recommended profile removes
that unused payload and stores each target MLA record as 512 E4M3 data bytes
plus four FP32 scales: **528 bytes per token per layer**. We also keep the
sparse indexer in FP8, sequence-shard the target cache with DCP2, and use an
exact-fit page layout so the target and DFlash2 caches share storage without a
large padding fallback. Compact KDA replay preserves only the recurrent state
needed to commit accepted draft tokens.

With two 128 GB DGX Sparks, `GPU_MEMORY_UTILIZATION=0.87`, and the 1M profile,
the canonical EXL3 + FP8 configuration normally exposes **about 4.6–4.7
million logical KV tokens**—roughly 4.4–4.5 completely full 1M contexts. This
is aggregate logical capacity after DCP2; available bytes are still bounded by
the lower-memory rank. Unified-memory and page-cache state can move the exact
number between boots, so use the capacity printed during startup as the
authoritative value.

For maximum capacity, `exl3-fp4-dcp2` uses our 288-byte native-NVFP4 MLA
record, four-over-six scale selection, and calibrated per-layer outer scales.
It has exposed about **7.8 million logical KV tokens** on the same pair. We
still recommend FP8 KV for general use because it keeps more numerical
headroom and produced the better overall quality/parallel-decode balance; the
FP4 profile is there for workloads where capacity matters more.

## Requirements

- Two ARM64 NVIDIA DGX Spark systems with Docker and a working NVIDIA container
  runtime.
- Passwordless SSH from the head node to the worker.
- A working ConnectX RoCE/IB path between the nodes.
- `git`, `curl`, `ssh`, `rsync`, and the Hugging Face CLI on the head.
- Enough storage on both systems for the approximately 164 GiB EXL3 target,
  the drafter, the runtime image, and compilation caches.

Run the commands below on the **head node**. The worker does not need its own
Git checkout: the launcher copies its runtime files and, when necessary, the
Docker image.

On the 128 GB unified-memory systems, keep swap available but set
`vm.swappiness=0`. Check this again after a reboot.

## Quick start

### 1. Clone and install the download helper

```bash
git clone https://github.com/coolbho3k/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark.git
cd GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark
python3 -m pip install --user -U 'huggingface_hub[cli]'
```

The pinned checkpoints are public. If you use a Hugging Face token, export it
as `HF_TOKEN` before running the preparation step.

### 2. Configure the cluster

The checked-in defaults describe our `spark-0` + `dgx1.lan` cluster. For a
different pair, export the worker hostname, node IPs, and fabric interfaces:

```bash
export WORKER_HOST=dgx1.lan
export HEAD_IP=10.100.32.1
export WORKER_IP=10.100.32.2
export NCCL_IB_HCA=rocep1s0f1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_ADDR_RANGE=10.100.32.0/24
```

Use `ibdev2netdev` and `ip -br addr` to find the correct interface names and
addresses. Both IPs must be reachable over the selected interface, and the
same absolute model paths must be usable on both nodes.

Inspect the fully resolved serving configuration without launching anything:

```bash
./serve-profile.sh show exl3-fp8-dcp2
```

### 3. Download, build, and start

```bash
./serve-profile.sh prepare exl3-fp8-dcp2
./serve-profile.sh build exl3-fp8-dcp2
./serve-profile.sh start exl3-fp8-dcp2
```

`prepare` downloads pinned target and draft revisions to the head and rsyncs
them to the worker. `build` creates the patched image on the head. `start`
copies the image to the worker if needed, verifies the runtime files on both
ranks, starts the worker and then the head, and waits for `/health`.

Cold model startup normally takes several minutes. The command returns only
after the API is healthy or startup has failed.

#### Optional tuned EXL3 path

Rebuild from this checkout before enabling the tuned decode kernel and fused
prefill activation; older images may not contain the required native extension:

```bash
./serve-profile.sh build exl3-fp8-dcp2
GLM53_EXL3_MOE_FAST=1 EXL3_FUSED_FAT_ACTIVATION=1 \
  TARGET_CUDAGRAPH_SCOPE=c1 ./serve-profile.sh start exl3-fp8-dcp2
```

This keeps the same weight and KV formats and adds target decode graphs for
C1; drafter graphs still cover C1–C6. The kernel settings are opt-in, require
a restart, and appear in `serve-profile.sh show` when supplied as environment
overrides. Startup verifies the EXL3 native binary on both nodes, not just
the Python files. See the [serving campaign notes](docs/SERVING-CAMPAIGN-20260904.md)
for the measured gains and validation scope.

### 4. Call the API

The default endpoint is `http://<head-ip>:8000/v1`, and the served model name
is `glm-5.3-flash`.

```bash
curl -fsS http://10.100.32.1:8000/health

curl http://10.100.32.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5.3-flash",
    "messages": [{"role": "user", "content": "Write a haiku about Blackwell."}],
    "reasoning_effort": "high",
    "max_tokens": 256
  }'
```

The server does not configure API authentication. Keep it on a trusted network
or put an authenticated reverse proxy in front of it before exposing it.

### 5. Stop the cluster

```bash
./serve-profile.sh stop
```

This removes the serving container from both nodes. The downloaded weights,
Docker image, and compilation caches remain for the next launch.

## Common configuration

All serving settings are environment overrides. Export them once in the shell
before `prepare`, `build`, or `start`, or prefix an individual command.

| Variable | Default | Purpose |
|---|---:|---|
| `WORKER_HOST` | `dgx1.lan` | SSH hostname for rank 1 |
| `HEAD_IP` | `10.100.32.1` | Rank 0 distributed/API address |
| `WORKER_IP` | `10.100.32.2` | Rank 1 distributed address |
| `NCCL_IB_HCA` | `rocep1s0f1` | RoCE/IB device |
| `NCCL_SOCKET_IFNAME` | `enp1s0f1np1` | Network interface used by NCCL/Gloo |
| `NCCL_IB_ADDR_RANGE` | `10.100.32.0/24` | Allowed RoCE address range |
| `API_PORT` | `8000` | OpenAI-compatible API port |
| `GPU_MEMORY_UTILIZATION` | `0.87` | Fraction of unified memory available to vLLM |
| `MAX_MODEL_LEN` | `1048576` | Per-request model context limit |
| `MAX_NUM_SEQS` | `6` | Maximum concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Pure-prefill scheduler budget and EXL3 scratch size |
| `MODEL_HOST_PATH` | profile-specific | Target checkpoint path on both nodes |
| `DRAFT_HOST_PATH` | profile-specific | DFlash2 checkpoint path on both nodes |

For example, to trade some concurrency for more system headroom:

```bash
GPU_MEMORY_UTILIZATION=0.84 MAX_NUM_SEQS=4 \
  ./serve-profile.sh start exl3-fp8-dcp2
```

Do not manually set vLLM's `--kv-cache-memory`. Let the startup profiler reserve
space for activation peaks. Also leave the recommended profile's block size
alone; its value is part of the exact-fit target/draft cache geometry.

## Available profiles

```bash
./serve-profile.sh list
```

| Profile | Target weights | Target KV | Default drafter | Default context |
|---|---|---|---|---:|
| **`exl3-fp8-dcp2`** | **EXL3/TR3 4 bpw** | **FP8** | **MXFP8** | **1M** |
| `exl3-fp4-dcp2` | EXL3/TR3 4 bpw | native FP4 | BF16 | 1M |
| `nvfp4-fp8-dcp2` | optimized NVFP4/Marlin | FP8 | MXFP8 | 512K |
| `nvfp4-fp4-dcp2` | optimized NVFP4/Marlin | native FP4 | BF16 | 1M |

The first row is the canonical recommendation. The other profiles remain
available for controlled quality, speed, and capacity comparisons. Use the
same three commands with another profile name:

```bash
./serve-profile.sh prepare nvfp4-fp4-dcp2
./serve-profile.sh build nvfp4-fp4-dcp2
./serve-profile.sh start nvfp4-fp4-dcp2
```

On a fresh Docker cache, build `nvfp4-fp4-dcp2` before
`exl3-fp4-dcp2`; the latter layers its native-FP4 support on that image chain.
The MXFP8 drafter is currently packaged only in the FP8-KV image.

To use the original BF16 drafter with the recommended target:

```bash
DFLASH_DRAFT_VARIANT=bf16 ./serve-profile.sh prepare exl3-fp8-dcp2
DFLASH_DRAFT_VARIANT=bf16 ./serve-profile.sh start exl3-fp8-dcp2
```

The optimized NVFP4 profiles download the published
[coolbho3k/GLM-5.3-Flash-NVFP4-Optimized](https://huggingface.co/coolbho3k/GLM-5.3-Flash-NVFP4-Optimized)
checkpoint; rebuilding its quantization is not required to serve it.

## Scheduler and CUDA-graph controls

The recommended adaptive scheduler gives pure-prefill work the full 8,192-token
budget. When decode is active, it admits smaller aggregate prefill packets at a
cadence based on the number of active decoders. This keeps ongoing chats and
agents responsive without completely starving new prompts.

Its policy is in [`scheduler_profiles/adaptive.json`](scheduler_profiles/adaptive.json).
The scheduler checks the file for updates once per second and retains the last
valid configuration if an edit is malformed. See
[`scheduler_profiles/README.md`](scheduler_profiles/README.md) for each field.

Useful switches:

```bash
# Previous fixed 512-token/every-4-step policy
PREFILL_ADMISSION_POLICY=static ./serve-profile.sh start exl3-fp8-dcp2

# Unmodified vLLM scheduling
ENABLE_DECODE_FIRST_SCHEDULER=0 ./serve-profile.sh start exl3-fp8-dcp2

# Optional target CUDA graph for C1; may improve C1 decode but uses KV headroom
TARGET_CUDAGRAPH_SCOPE=c1 ./serve-profile.sh start exl3-fp8-dcp2
```

The default already enables private drafter graphs for C1–C6. The target stays
eager because broad target graph capture did not improve this stack. Piecewise
graphs remain a diagnostic option in
[`docs/EXL3-FP8-DCP2.md`](docs/EXL3-FP8-DCP2.md), not a recommended default.

## Reasoning effort

The API defaults to `high`. GLM-5.3 exposes three native thinking levels, so
common OpenAI values are mapped as follows:

| API value | GLM mode |
|---|---|
| `none` | thinking disabled |
| `minimal`, `low` | `low` |
| `medium`, `high` | `high` |
| `xhigh`, `max` | `max` |

For Chat Completions, pass the top-level `reasoning_effort` field shown above.
For the Responses API, use `"reasoning": {"effort": "medium"}`. Output-token
limits include reasoning tokens when thinking is enabled.

## Operations and troubleshooting

- Check liveness with `/health`, not `/v1/models`; the model-list endpoint can
  return before the distributed engine is usable.
- Follow rank-0 logs with `docker logs -f vllm_glm53` and rank-1 logs with
  `ssh "$WORKER_HOST" docker logs -f vllm_glm53`.
- Use `serve-profile.sh start`; it removes stale containers from both nodes
  before forming a new rendezvous and cleans up both ranks if startup fails.
- A wrong HCA, interface, or subnet can look like a model startup hang. Verify
  the fabric settings first when the ranks cannot join.
- The launcher verifies runtime-defining files across ranks and replaces a
  missing or stale worker image for the recommended profile.
- Model loading is intentionally conservative. The experimental direct-I/O
  loader was much faster but unstable in multi-node operation on this hardware.

## Further reading

- [Recommended EXL3 + FP8 recipe](docs/EXL3-FP8-DCP2.md)
- [DFlash2 integration](docs/DFLASH2-SPECULATIVE-DECODING.md)
- [Adaptive scheduler results and rationale](docs/ADAPTIVE-SCHEDULER-RESULTS.md)
- [NVFP4 weight optimization](docs/NVFP4-WEIGHT-OPTIMIZATION.md)
- [Native-FP4 KV optimization](docs/KV-OPTIMIZATION-STATUS.md)
- [Deployment fixes and full serve arguments](docs/DEPLOY-REPORT.md)
- [Known open problems](docs/OPEN-PROBLEMS.md)

## Credits

[Z.ai](https://huggingface.co/zai-org) for GLM-5.3-Flash,
[Mia-AI Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
for the EXL3 checkpoint and integration work,
[local-inference-lab](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8)
and [incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) for the DFlash2
checkpoints, [Red Hat AI](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
for the NVFP4 foundation, and the vLLM, FlashInfer, ExLlamaV3, and SparkInfer
projects for the serving kernels and runtime.
