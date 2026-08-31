# KV optimization: final implementation report

This report is the acceptance record for
`/home/emi/code/glm-kv-opt.md`. Measurements are from the two-node
`spark-0` + `dgx1.lan` GB10 cluster on 2026-08-31. “Capacity” below means
physical cache capacity; the production launcher deliberately keeps
`--max-num-seqs 6` as its serving-performance setting.

## Long-context production addendum

The final two fixes remove request-length-dependent artificial limits without
changing the native-FP4 record geometry:

- vLLM's global block-table constructor multiplied all hybrid groups by DCP2.
  That is correct for sequence-sharded target MLA, but it halved the tables for
  replicated KDA and TP-head-sharded/sequence-replicated DFlash state. The old
  image therefore indexed past the Mamba table near 133K tokens even with most
  of the KV pool free. Replicated groups now use their physical table width.
- the sparse indexer reserved `40 * max_model_len` workspace entries even
  though GLM stores one index entry per four-token k-pool. The bound is now
  `ceil(max_model_len / index_kpool)`, with the existing chunk splitter
  serializing multiple enormous prefills when necessary.

Measured final configuration:

| Max request length | KV capacity | Validation |
|---:|---:|---|
| 262,144 | **at least 2,198,799 tokens** | 252,049-token retrieval and 235,520-token prefix hit passed |
| 1,048,576 | **3,561,829 tokens** (428 blocks, 3.40x) | 1,020,049-token retrieval and 1,003,520-token prefix hit passed |

The uncached 1,020,049-token request completed in 723.817 seconds, about 1,409
input tok/s. Reusing its 1,003,520-token prefix completed in 14.718 seconds.
The two-round, 400-token decode sweep on the 1M configuration produced 33.4
tok/s at C1 and 55.4 aggregate tok/s at C6, with zero failures. Relative to the
256K configuration that is 8.5% lower at C1 and 1.6% lower at C6; the capacity
fixes did not introduce a large normal-serving penalty.

## 65K optimization baseline

The 65K baseline used `glm53-v11:kvopt-final`. It preserves DFlash2 K=7,
text, tool calling, prefix caching, chunked prefill, image input, and video
input. Its production layout is:

- TP2 plus DCP2 over RoCE;
- 288-byte/token/layer GLM-native zero-RoPE NVFP4 sparse-MLA records;
- 17-byte/token/layer native SM121 MXFP4 sparse-indexer records;
- global DCP sparse top-k with correct page-ownership conversion;
- FP16 persisted KDA state and compact accepted-prefix replay instead of eight
  speculative state snapshots;
- replicated KDA and TP-head-sharded, sequence-replicated DFlash state; and
- 5,120-token manager pages with exact-fit 2,208-token DFlash pages.

The launcher's `--kv-cache-dtype fp8_e4m3` remains as vLLM's safe fallback
for cache types that do not select a specialized layout. Matching GLM sparse
MLA and indexer layers are explicitly replaced by the native FP4 specs and
kernels; their measured physical records are 288 and 17 bytes, respectively.

The limiting rank currently has 437 global pool blocks:

| Metric | Production value |
|---|---:|
| Bytes per global pool block | 17,177,600 |
| MLA portion | 16,220,160 |
| Indexer/tail portion | 957,440 |
| Manager tokens per block | 5,120 |
| Physical bytes per manager token | 3,355 |
| DCP-global logical capacity | **954,641 tokens** |
| Fractional 65K contexts | **14.57** |
| Complete 65K physical reservations | **14** |

The pre-optimization FP8/DCP1 accounting boot held 166,525 logical tokens, or
2.54 65K contexts. The final result is 5.73x that logical capacity and exceeds
the plan's 850K–1.05M aggressive range. This is a measured layout/capacity
change, not a larger `--max-model-len` headline.

## Physical accounting

The profiler identity used for each rank is:

```text
requested = weights + persistent_non_weight + transient_peak + available_KV
```

It closes with zero-byte identity error on both ranks:

| Component | Rank 0 | Rank 1 (limiting) |
|---|---:|---:|
| Requested by `gpu-memory-utilization=0.87` | 105.816 GiB | 105.870 GiB |
| Model weights and retained model storage | 90.606 GiB | 90.606 GiB |
| Persistent non-weight/runtime allocation | 3.903 GiB | 4.454 GiB |
| Transient activation peak | 3.814 GiB | 3.814 GiB |
| Available KV budget | 7.493 GiB | 6.997 GiB |
| Identity error | 0 B | 0 B |

Rank 1 retains about 0.551 GiB more non-weight runtime memory on this boot and therefore
sets the common pool. Model geometry, transient peak, and bytes per cache block
are identical between ranks. Earlier 391K/461K-class variation came from UMA
free-memory and runtime allocation differences at profile time; it was not a
different cache layout. The allocator tail on today's limiting rank is
5,953,291 bytes.

The resident dtype inventory covers 99.883% of the profiler's model-memory
number (97,173,816,288 inventoried bytes versus 97,287,399,936 bytes). The
representative-rank inventory is:

| Storage dtype | Bytes |
|---|---:|
| INT32 packed/quantized storage | 76,101,453,544 |
| BF16 | 11,410,398,208 |
| FP8 | 9,512,681,472 |
| FP32 | 149,283,064 |

## Capacity by request length

Hybrid caches reserve whole component pages. These numbers account separately
for target MLA, the indexer tail, replicated KDA state, and the bounded DFlash
window:

| Request | Pool blocks/request | Physical/request | Fractional fit | Whole fits | Effective logical capacity |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 13 | 0.208 GiB | 33.62 | 33 | 68,844 |
| 8,192 | 17 | 0.272 GiB | 25.71 | 25 | 210,582 |
| 32,768 | 27 | 0.432 GiB | 16.19 | 16 | 530,356 |
| 65,536 | 30 | 0.480 GiB | **14.57** | **14** | **954,641** |

At 65K, target-page rounding wastes 6,144 logical token slots per request and
the DFlash window wastes 1,729. A mixed portfolio containing one request at
each listed length consumes 87 blocks and 108,544 logical tokens. Five complete
portfolios fit: 20 physical reservations and 542,720 request tokens; the
fractional packing metric is 545,215 logical tokens. The live
server's `--max-num-seqs 6` limits simultaneously scheduled sequences even
when more physical reservations fit.

## Stage results

### 1. Accounting and instrumentation

Structured log records cover model load/dtype inventory, persistent runtime
growth, activation profiling, cache specs/groups, tensor sharing, physical
page sizes, allocator tails, and final capacity. The measured identities close
exactly and the report derives the same 437 blocks and 954,641 slots vLLM
reports. Run:

```bash
(docker logs vllm_glm53 2>&1; \
 ssh -o BatchMode=yes dgx1.lan 'docker logs vllm_glm53 2>&1') \
  | python3 probes/kv_accounting_report.py
```

### 2. Zero-RoPE FP8

The strict GLM `qk_rope_head_dim=0` path removes the generic, zero-filled RoPE
payload. Generic geometry retains its compatibility layout. This remains the
FP8 rollback/reference implementation.

### 3. Hybrid allocation geometry

The cache-spec and hybrid-manager patches compute real page requirements and
allow exact-fit sharing across heterogeneous MLA, KDA, indexer, and DFlash
groups. Short, mixed, and longest-request numbers are reported separately
above; no short-packing result is presented as 65K capacity.

### 4. DCP2

Target MLA and indexer token storage are sequence-sharded across the two TP
ranks. KDA remains replicated. DFlash has four distinct TP-local KV heads, so
its short sliding-window cache is sequence-replicated rather than incorrectly
DCP-sharded. FlashInfer's DCP wrapper performs query exchange and distributed
LSE/output merging, including DFlash's non-causal verification block.

The isolated two-rank NCCL test used the production RoCE HCA and interface:

| Representative collective | Payload/rank | Mean | p95 | Effective bus GiB/s |
|---|---:|---:|---:|---:|
| Indexer candidate all-gather | 96 KiB | 90.7 µs | 441.8 µs | 1.01 |
| DCP query all-to-all | 1.5 MiB | 150.6 µs | 344.8 µs | 4.86 |
| DCP output all-reduce | 1.5 MiB | 438.6 µs | 461.6 µs | 3.34 |

The backend introduces no extra host-side synchronization in the serving hot
path. These are representative collective microbenchmarks, not a claim that
every token executes each exact PyTorch operation. Their sub-millisecond p95s
show that RoCE is not the dominant serving latency.

### 5. Native FP4 sparse MLA

Both required layouts were implemented and separately exercised:

| Layout | Record | Limiting blocks | 65K logical capacity | Disposition |
|---|---:|---:|---:|---|
| 5A compatible FP4 + retained RoPE field | 368 B | 281 | 736,624 (11.24x) | Experimental |
| 5B GLM-native FP4, no RoPE field | **288 B** | **437** | **954,641 (14.57x)** | Production |

The isolated layout probe found zero mismatched latent bytes and exactly equal
attention output/LSE. Native storage is 21.739% smaller, and it was marginally
faster in this microbenchmark: 0.079 vs 0.087 ms write and 0.234 vs 0.241 ms
read.

5A passed text, tool, PNG, video, 27K retrieval, ordinary prefix reuse, and its
C1/C3/C6 throughput checks. It is not promoted because divergent concurrent
requests sharing one long prefix reported zero reused tokens, despite correct
answers. Native 5B passes that same concurrency gate with 40,960 hit tokens.

### 6. FP4 indexer

The SM121 writer, paged reader, DCP ownership map, global scorer merge, and
split physical page layout are implemented without full-cache BF16
materialization. Final isolated gates:

- 132-byte FP8 and 68-byte FP4 gather layouts: zero byte mismatches;
- FP4 writer to vLLM gather: values and scales byte-exact;
- paged versus contiguous scorer: max absolute error 0;
- native scorer versus reference: max error 1.526e-5, mean 2.169e-6; and
- top-512 selection: 100% recall on all 8 rows.

### 7. Composition, quality, and performance

The full production validator passed:

- health and exact deterministic text;
- arithmetic result 1517;
- structured Seattle tool call;
- red PNG and red MP4 recognition;
- 27,049-token retrieval returning `ORCHID`;
- single-request prefix reuse with 10,240 hit tokens; and
- four divergent suffixes sharing a 24,045-token prefix, all returning
  `MAGNOLIA` with 40,960 aggregate hit tokens.

Two-round, 400-token C1–C6 results had zero failures:

| | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| Aggregate tok/s | 33.9 | 38.7 | 48.1 | 50.7 | 62.4 | 62.4 |
| Per-stream tok/s | 33.9 | 19.8 | 19.8 | 16.5 | 14.1 | 12.9 |
| Accepted/drafted | .489 | .462 | .424 | .391 | .460 | .413 |

The historical FP8 production control was 35.1, 41.6, 40.6, 47.5, 56.2, and
47.7 aggregate tok/s, with acceptance .525, .450, .420, .400, .510, and .400.
The final candidate is 3–7% lower at C1/C2 and 7–31% higher at C3–C6: serving
performance is retained while high-concurrency throughput improves.

## Packaging and reproduction

The long-context candidate was independently built and fully validated on each
ARM node:

- `spark-0`: `sha256:998d53a59ede9ea1e1f2e8ea5841ee9838fa709af4ac285d010bbebfc4f5e975`
- `dgx1.lan`: `sha256:a0ab2659fb82a7f8a579ad0641b8bc068b7acb9a082586b97bf54bdeb911d9a4`

`start-cluster.sh` verifies that the fifteen runtime files defining layout, DCP,
KDA replay, accounting, and SM121 kernels have identical SHA-256 hashes across
both images before it starts either rank.

The ground-up packaging entry point is `build-production-image.sh`. It pins the
public DFlash2 base by digest, checks out SparkInfer commit
`3a437ab5168060e4d625f05e1625c04089f1ba37`, applies the checked-in native-FP4
delta, and builds `overlay-dflash2/Dockerfile.reproducible`. No local experiment
image tag is an input:

```bash
./build-production-image.sh
```

Start and stop the validated deployment:

```bash
./start-cluster.sh
./stop-cluster.sh
```

Rollback choices:

```bash
# Compatible FP4 experiment (not recommended for prefix-sharing workloads)
IMAGE=glm53-v11:nvfp4-dcp2-kda-compact-compat5a ./start-cluster.sh

# Original FP8/DCP1/DFlash2 implementation
IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 \
  DCP_SIZE=1 USE_FP4_INDEXER_CACHE=0 ./start-cluster.sh
```

## Deferred BF16 projection

Per the specification, remaining BF16 model components were inventoried but
not quantized without a separate user decision. Idealized upper bounds are:

| Hypothetical conversion | Ideal bytes saved | Extra blocks | Projected 65K logical capacity |
|---|---:|---:|---:|
| BF16 to 8-bit | 5,705,199,104 | 332 | 1,679,906 |
| BF16 to 4-bit | 8,557,798,656 | 498 | 2,042,538 |

These deliberately assume zero scale/metadata, activation, kernel, quality, or
performance cost. Major BF16 residents include embeddings/LM head, draft RoPE
cache and draft FC, and KDA input projections. The figures are opportunity
bounds, not validated configurations or authorization to change model
precision.

## Limitations

- UMA/runtime state at profile time can change available blocks across cold
  boots; compare physical bytes/block and read both ranks. The verified 1M
  profile reported 3,561,829 logical tokens.
- The production default accepts 262,144 tokens per request; set
  `MAX_MODEL_LEN=1048576` for the validated model-native 1M mode. Both schedule
  at most six sequences concurrently under the production performance setting.
- 5A's concurrent shared-prefix accounting failure keeps it experimental.
- The final image tag is local to each node. The complete public source recipe
  rebuilds it from pinned inputs; model and draft weights remain external.
- Remaining BF16 conversion is only projected. It requires its own kernel,
  quality, multimodal, and performance acceptance cycle.
