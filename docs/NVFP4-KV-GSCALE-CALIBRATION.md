# NVFP4 KV outer-scale calibration tooling

Status: calibrated and validated on 2026-09-02. The cross-node-gated v1
artifact is checked in at `kv_calibration/results/nvfp4-gscale-v1.json`
and is the default for the v14 cluster launcher. Set
`USE_CALIBRATED_NVFP4_MLA=0` for the exact `G=1` rollback.

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

## Selected v1 result

The fit used 330 capture shards from both DCP ranks, 15 prefill/decode strata,
and 196 prompts. The final grid searched `G=1 ... 2` at 32 steps per
octave. Candidates were fit on the merged population, held out by stratum, and
then independently required not to regress on either node. Layer 43 failed the
independent worker gate and reverted to `G=1`.

| layer | G | reader L |
|---:|---:|---:|
| 3 | 1.090508 | 0.917004 |
| 7 | 1.114387 | 0.897355 |
| 11 | 1.000000 | 1.000000 |
| 15 | 1.138789 | 0.878126 |
| 19 | 1.414214 | 0.707107 |
| 23 | 1.383910 | 0.722590 |
| 27 | 1.090508 | 0.917004 |
| 31 | 1.414214 | 0.707107 |
| 35 | 1.044274 | 0.957603 |
| 39 | 1.414214 | 0.707107 |
| 43 | 1.000000 | 1.000000 |

Merged held-out MSE improves 0.24171%; independent head and worker holdouts
improve 0.16250% and 0.20658%. All seven distinct selected `L` values
produced exactly the CPU-reference E4M3 scale codes in the real SM121 writer;
GPU/reference MSE differed by at most `8.4e-9`. The serving gate passed
exact text, arithmetic, tools, vision, a 390,042-token retrieval, and a
378,880-token prefix-cache replay. The record remains exactly 288 bytes/token.

## Public corpus recipe

Start with NVIDIA ModelOpt's public calibration blend: `cnn_dailymail` plus
`nvidia/Nemotron-Post-Training-Dataset-v2`. Add NVIDIA RULER prompts at 32K,
128K, and 256K for retrieval, multi-hop, aggregation, and QA. Finally add code,
multilingual, and redacted deployment-representative chat. See
`kv_calibration/corpus-plan.example.json` for suggested counts.

The selected v1 corpus contains 48 natural-chat, 48 summarization, 32 code, 32
math-reasoning, and 30 multilingual-NLI prompts, plus three retrieval and three
aggregation prompts spanning approximately 32K, 128K, and 256K. Prompts are
calibration inputs only; reference answers are not captured or used to fit the
scale.

The checked-in public builder caches source rows, excludes reference answers,
and sizes the long prompts with the exact deployed tokenizer:

```bash
python3 -m kv_calibration.prepare_corpus \
  --cache-dir "$HOME/.cache/glm53-nvfp4-calibration/corpus-v1/sources" \
  --output "$HOME/.cache/glm53-nvfp4-calibration/corpus-v1/corpus.jsonl"
```

After the first download, add `--offline` for a network-independent rebuild.
The tokenizer endpoint must be healthy unless `--skip-long` is used.


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

## Capture

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
  --remote-control dgx1.lan:/same/absolute/path/control.json \
  --model MODEL_SERVED_BY_VLLM
```

After the corpus, the driver advances a two-node `flush_epoch` and sends one
one-token request. Each layer checkpoints its reservoir before sampling that
request, so even the final long-context stratum is durable. Use
`--no-final-flush` only when intentionally driving an older tooling image.

Capture copies only a bounded sample of 16-value groups to CPU. It is an
explicit calibration mode and can reduce serving performance; do not enable it
for normal traffic. Shards are atomically checkpointed and are mergeable across
both ranks and worker processes by their random reservoir priorities.

## Fit

```bash
python3 -m kv_calibration.fit \
  /path/to/captures-head /path/to/captures-worker \
  --output /path/to/nvfp4-gscale.json \
  --objective mse \
  --log2-global-min 0 --log2-global-max 1 \
  --steps-per-octave 32 \
  --per-stratum-limit 8192 \
  --cross-validation-root /path/to/captures-head \
  --cross-validation-root /path/to/captures-worker \
  --minimum-cross-validation-improvement-percent 0
```

The selected recipe canonicalizes power-of-two-equivalent scale phases to
`G >= 1`, balances bounded strata, reserves 20% held out, and falls back to
`G=1` if a layer regresses on either the merged holdout or an independent
capture root. The broader default grid remains useful for exploration.
Alternate objectives:

```bash
python3 -m kv_calibration.fit /path/to/captures \
  --output /tmp/gscale-relative.json --objective group-relative
python3 -m kv_calibration.fit /path/to/captures \
  --output /tmp/gscale-blend.json --objective blend \
  --blend-relative-weight 0.10
```

## Serve the validated artifact

Build v14 on both nodes. The checked-in cluster wrapper copies the selected
artifact to the worker and mounts the local copy into the head automatically:

```bash
./build-gscale-tooling-image.sh
./start-cluster.sh
```

To serve another fit:

```bash
IMAGE=glm53-v14:nvfp4-gscale-tooling \
NVFP4_CALIBRATION_HOST_PATH="$HOME/.cache/glm53-nvfp4-calibration/custom" \
NVFP4_MLA_SCALES_FILE=nvfp4-gscale.json ./start-cluster.sh
```

Strict mode rejects a malformed schema, wrong algorithm/layout, invalid scale,
or a missing MLA layer entry during model initialization. Writer and reader use
the same `L`. Roll back only the outer scales with:

```bash
USE_CALIBRATED_NVFP4_MLA=0 ./start-cluster.sh
```

## Completed validation

- CPU fit: merged and per-node held-out non-regression gates passed.
- GPU writer: every distinct selected scale matched all 12,288 reference scale
  codes; MSE parity passed.
- Capacity: the 288-byte layout and accounting are unchanged. A temporary 0.85
  validation boot exposed 2,063,863 logical tokens (1.97x at 1M); UMA state
  controls the exact number of allocated blocks.
- End to end: all short functional and 390K retrieval/prefix cases passed.
- Serving: the first matched two-request C1 run measured 35.6 tok/s versus 36.2
  tok/s at `G=1`. The stochastic benchmark's acceptance varies enough that
  this is treated as no isolated speed claim; the static scale adds no record
  bytes or additional kernel stage.
