# NVFP4 KV outer-scale calibration tooling

Status: tooling candidate only. It has not been used to capture the model,
fit scales, or launch a calibrated server. The known-good v13 server remains
the production state.

## What it calibrates

The 288-byte record stays unchanged:

- 256 bytes: 512 signed E2M1 latent values
- 32 bytes: one E4M3 scale for each group of 16 values

The v13 writer already chooses `amax/6` or `amax/4` independently per group.
This tooling adds one positive outer scale `L` per MLA layer. The writer stores
`E4M3(local_scale / L)` and the reader reconstructs
`E2M1 * E4M3 * L`. The report also calls `G = 1/L` the global scale.
Neither `L` nor `G` occupies KV memory, so capacity remains exactly 288 bytes
per cached token for these layers.

MSE is the default fit objective. Dataset-level NMSE has the same optimum
because its signal-energy denominator is independent of `G`. Mean per-group
relative MSE is available as an alternate diagnostic/objective, but it gives
quiet groups disproportionate weight. Every report includes MSE, NMSE,
mean/p95/p99 group-relative MSE, `/4` selection rate, zero-scale rate, and
saturation rate.

## Public corpus recipe

Start with NVIDIA ModelOpt's public calibration blend: `cnn_dailymail` plus
`nvidia/Nemotron-Post-Training-Dataset-v2`. Add NVIDIA RULER prompts at 32K,
128K, and 256K for retrieval, multi-hop, aggregation, and QA. Finally add code,
multilingual, and redacted deployment-representative chat. See
`kv_calibration/corpus-plan.example.json` for suggested counts.

The corpus driver accepts JSONL rows containing either:

```json
{"bucket":"nemotron-natural","messages":[{"role":"user","content":"..."}]}
```

or:

```json
{"bucket":"ruler-retrieval","prompt":"...","max_tokens":64}
```

Use a separate held-out split during fitting. For a final go/no-go decision,
also compare deterministic end-to-end generations and the GPU parity probe;
the CPU fit emulates the formats exactly but not the writer's approximate
reciprocal instruction at every rounding boundary.

## Build only (safe while v13 is live)

```bash
./build-gscale-tooling-image.sh
```

This creates `glm53-v14:nvfp4-gscale-tooling`. Merely building it does not
touch the live server. With no calibration environment variables, its runtime
defaults to `L=1` and capture disabled.

## Capture (do not run until approved)

The future capture launch must mount a host directory at the same path on all
ranks and set:

```bash
VLLM_NVFP4_MLA_CAPTURE_DIR=/calibration/captures
VLLM_NVFP4_MLA_CAPTURE_CONTROL=/calibration/control.json
```

Optional limits (defaults shown):

```bash
VLLM_NVFP4_MLA_CAPTURE_GROUPS_PER_STRATUM=65536
VLLM_NVFP4_MLA_CAPTURE_GROUPS_PER_CALL=1024
VLLM_NVFP4_MLA_CAPTURE_FLUSH_EVERY=128
VLLM_NVFP4_MLA_CAPTURE_DECODE_THRESHOLD=64
```

The checked-in two-node launcher exposes the common settings without changing
its defaults:

```bash
IMAGE=glm53-v14:nvfp4-gscale-tooling \
NVFP4_CALIBRATION_HOST_PATH="$HOME/.cache/glm53-nvfp4-calibration" \
ENABLE_NVFP4_MLA_CAPTURE=1 ./start-cluster.sh
```

Then drive requests sequentially so the sidecar control label is unambiguous:

```bash
python3 -m kv_calibration.drive_capture corpus.jsonl \
  --control /path/shared-with-server/control.json \
  --model MODEL_SERVED_BY_VLLM
```

Capture copies only a bounded sample of 16-value groups to CPU. It is an
explicit calibration mode and can reduce serving performance; do not enable it
for normal traffic. Shards are atomically checkpointed and are mergeable across
both ranks and worker processes by their random reservoir priorities.

## Fit (do not run until capture is approved and complete)

```bash
python3 -m kv_calibration.fit /path/to/captures \
  --output /path/to/nvfp4-gscale.json \
  --objective mse
```

The default grid searches `G=2^-4 ... 2^16` in 1/8-octave increments, balances
the bounded strata, reserves 20% held out, and falls back to `G=1` for a layer
if its chosen scale regresses on held-out data. Alternate exploratory fits:

```bash
python3 -m kv_calibration.fit /path/to/captures \
  --output /tmp/gscale-relative.json --objective group-relative
python3 -m kv_calibration.fit /path/to/captures \
  --output /tmp/gscale-blend.json --objective blend \
  --blend-relative-weight 0.10
```

## Serve a validated artifact (do not run until approved)

Mount the JSON into both ranks and set:

```bash
VLLM_NVFP4_MLA_SCALES_FILE=/calibration/nvfp4-gscale.json
VLLM_NVFP4_MLA_SCALES_STRICT=1
```

With the project launcher, place the artifact under the same absolute host
directory on both nodes and use its filename:

```bash
IMAGE=glm53-v14:nvfp4-gscale-tooling \
NVFP4_CALIBRATION_HOST_PATH="$HOME/.cache/glm53-nvfp4-calibration" \
NVFP4_MLA_SCALES_FILE=nvfp4-gscale.json ./start-cluster.sh
```

Strict mode rejects a malformed schema, wrong algorithm/layout, invalid scale,
or a missing MLA layer entry during model initialization. Writer and reader use
the same `L`. No environment variable means the known-good `L=1` behavior.

## Required validation before keeping it

1. Compare CPU emulation to the GPU writer/reader at every selected layer scale.
2. Require held-out MSE/NMSE improvement with acceptable relative-tail metrics.
3. Boot at 1M context and verify the exact 288-byte accounting/KV capacity.
4. Run deterministic short, coding, multilingual, 32K/128K/256K RULER, and
   long-chat continuation checks against v13.
5. Benchmark warmed C1 decode, prefill, and mixed load. A static outer scale is
   not expected to impose meaningful steady-state overhead, but it must be
   measured rather than assumed.
