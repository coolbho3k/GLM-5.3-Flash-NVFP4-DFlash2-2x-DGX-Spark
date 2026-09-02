# FlashInfer B12X W4A16 MoE A/B

This is an opt-in serving-kernel experiment. It changes only the routed-expert
MoE backend from Marlin to FlashInfer B12X W4A16. It does not change the model
weights, native-FP4 KV format, calibrated KV outer scales, FP8 indexer cache,
DFlash2 configuration, DCP2 topology, scheduler, context length, or memory
utilization.

The derived image contains three compatibility/safety fixes already exercised
in the earlier scratch campaign:

1. static expert-map support for W4A16 expert parallelism;
2. GLM's SwiGLU clamp passed through the B12X adapter and capability oracle;
3. eager, shape-cached workspaces by default, avoiding the previously rejected
   unbounded 4,096-token per-layer CUDA-graph allocation.

## Prepare without touching the server

```bash
./build-b12x-w4a16-image.sh
```

This builds `glm53-v15:b12x-w4a16-ab` on the head and `dgx1.lan`, verifies
the patch interfaces during the build, and checks that the installed runtime
files are byte-identical. It does not launch a container.

## Run after explicit authorization

Capture the already-running Marlin baseline first:

```bash
./run-b12x-w4a16-ab.sh baseline
```

Then replace it with the candidate. The explicit guard prevents accidental
restarts:

```bash
ALLOW_SERVER_RESTART=1 ./run-b12x-w4a16-ab.sh start-b12x
./run-b12x-w4a16-ab.sh candidate
./run-b12x-w4a16-ab.sh compare
```

Restore Marlin if B12X is not a clear winner:

```bash
ALLOW_SERVER_RESTART=1 ./run-b12x-w4a16-ab.sh restore-marlin
```

The benchmark uses deterministic, distinct prefixes and streaming responses.
It records:

- cold 8K, 32K, and 128K TTFT/input throughput;
- decode-only throughput;
- decode throughput and stream-gap behavior while a 32K cold prefill arrives;
- the concurrent prefill's TTFT/input throughput.

The comparison requires a meaningful geometric-mean prefill gain while
rejecting material decode regressions. It reports a recommendation but never
promotes or restores a backend automatically.

## Safety notes

- Keep `VLLM_B12X_USE_CUDA_GRAPH=0`. The eager workspace path is the safe
  configuration and the only one in this A/B.
- Do not run the CUDA EP microtest while the serving process owns nearly all
  memory. The full two-node startup is the integration gate. The small test is
  installed at `/opt/glm53-tests/test_b12x_w4a16_ep.py` for isolated use.
- B12X must be visible in the startup log as
  `Using 'FLASHINFER_B12X' NvFp4 MoE backend`; the controller checks this
  before candidate benchmarking.
- Both benchmark sides use the same final checkpoint and
  `gpu_memory_utilization=0.87`.

## What the long-context correctness probe covers

The retrieval probe writes and reads native-FP4 KV through all sparse-attention
layers and checks prefix-cache reuse. It is a useful end-to-end correctness
gate, not a capacity or statistical-accuracy proof: it is one sequence with
one sentinel and very little decode. The campaign's prior 390K retrieval and
378,880-token replay remain the stronger long-context result. A final candidate
must additionally pass the same retrieval probe after startup.
