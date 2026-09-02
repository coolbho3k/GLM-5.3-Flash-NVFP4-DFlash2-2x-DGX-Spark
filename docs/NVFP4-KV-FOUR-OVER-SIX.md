# NVFP4 KV four-over-six writer

The default `glm53-v13:nvfp4-four-over-six` image ports vLLM's
`nvfp4_4over6` store-time scale search into B12x's native 288-byte GLM MLA
writer. It is based on the [upstream vLLM CUDA
implementation](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu)
and the [NVIDIA NVFP4 KV-cache
description](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/).

## What changes

For every 16-value compressed-latent group, the writer constructs two legal
E4M3 block-scale candidates, `amax/6` and `amax/4`. It quantizes the values to
E2M1 under both candidates, reconstructs them, computes SSE, and stores the
`/4` candidate only when its error is strictly lower. Ties retain `/6`.

The implementation evaluates the actual packed SM121 E2M1 codes with native
pack/decode instructions and writes the already-computed winner. The cache ABI
does not change:

```text
[  0, 256)  512 packed E2M1 latent values
[256, 288)   32 E4M3 group-scale bytes
```

Readers, attention kernels, capacity accounting, prefix-cache identity, and
the 288-byte record width are unchanged. The extra work occurs only when a
token is written; reading an already-cached token has no new work. The kernel
compile version is bumped so a cached amax/6 cubin cannot be reused.

## Reproduce

Build the three-image chain on both ARM64 nodes:

```bash
./build-production-image.sh
./build-fp8-passthrough-image.sh
./build-four-over-six-image.sh
```

The last command creates `glm53-v13:nvfp4-four-over-six`. Start it from the
head node:

```bash
./start-cluster.sh
```

Roll back only the writer search, leaving the checkpoint and all other serving
settings unchanged:

```bash
IMAGE=glm53-v12:fp8-passthrough ./start-cluster.sh
```

## GPU validation on GB10/SM121

The standalone probe used identical deterministic BF16 values in both images:
384 tokens, 32 groups/token, 16 values/group, and the production 2304 block
size. Results over 12,288 groups:

| metric | amax/6 | four-over-six | change |
|---|---:|---:|---:|
| mean element MSE | 0.0890712 | 0.0788473 | -11.48% |
| mean group SSE | 1.4251385 | 1.2615563 | -11.48% |
| E4M3-zero groups | 1,261 | 766 | -39.25% |
| payload SHA-256 | `c09cc746…ab536` | `30fe828f…0e8d` | changed as expected |

Four-over-six changed 4,449 scale bytes (36.2%). At a `1e-7` SSE tolerance,
4,446 groups improved, 7,842 tied, and **zero regressed**; total group SSE fell
by 2,010.10.

The 1,000-iteration writer timings were launch-bound and showed no measurable
penalty:

| tokens/write | amax/6 | four-over-six |
|---:|---:|---:|
| 1 | 0.07512 ms | 0.07482 ms |
| 8 (DFlash K=7 verification) | 0.07455 ms | 0.07411 ms |
| 384 | 0.07405 ms | 0.07406 ms |

The existing 368-byte compatibility versus 288-byte native-layout test also
passed with bit-identical latent payloads, exact attention output/LSE equality,
and a 0.07730 ms native write.

Run the numerical/timing probe in both images with:

```bash
docker run --rm --gpus all --ipc host --entrypoint python3 \
  -v "$PWD/probes/probe_nvfp4_four_over_six.py:/workspace/probe.py:ro" \
  glm53-v12:fp8-passthrough /workspace/probe.py --label amax6

docker run --rm --gpus all --ipc host --entrypoint python3 \
  -v "$PWD/probes/probe_nvfp4_four_over_six.py:/workspace/probe.py:ro" \
  glm53-v13:nvfp4-four-over-six /workspace/probe.py --label four_over_six
```

The first full 1M-profile boot reported 3,270,558 logical KV tokens. The
existing two-round 400-token C1 harness returned 37.2 tok/s, 50.4% accepted
draft tokens, and zero failures. These serving numbers rule out a material
regression but should not be interpreted as an isolated speedup because that
harness salts its prompts.

The deterministic behavioral suite passed every substantive case, including
both long-context retrieval repeats. Two original checks exhausted their tiny
96/16-token budgets inside the intentionally enabled default-high reasoning;
with 256-token budgets they returned the normal `test` response and exact
`ORCHID` output.
