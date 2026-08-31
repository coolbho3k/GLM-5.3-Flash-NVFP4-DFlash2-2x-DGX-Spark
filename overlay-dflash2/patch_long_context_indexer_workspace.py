#!/usr/bin/env python3
"""Replace the DeepSeek-era 40x indexer workspace bound for GLM k-pooling.

The generic sparse-indexer helper reserves ``40 * max_model_len`` key rows so
many DeepSeek prefill requests can share one flattened workspace. GLM's
indexer stores one row per complete ``index_kpool`` token pool, and the
existing request chunker already splits a group before its total key rows
exceed this helper's return value. Consequently the required correctness
bound is one maximum-length request after compression, not forty uncompressed
requests. Keeping the generic fallback unchanged limits this correction to
models that explicitly declare ``index_kpool > 1``.
"""

from pathlib import Path


TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"
)
MARKER = "GLM-KPOOL-WORKSPACE-BOUND"

OLD = '''def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40
'''

NEW = '''def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    index_kpool = int(
        getattr(vllm_config.model_config.hf_config, "index_kpool", 1) or 1
    )
    if index_kpool > 1:
        # GLM-KPOOL-WORKSPACE-BOUND: compressed_seq_lens, cache gathering and
        # sparse logits are pool-granular. split_indexer_prefill_chunks groups
        # requests only while their summed compressed lengths fit this bound;
        # when a single request reaches the bound it is accepted and its query
        # dimension is split by the separate logits cap. One maximum request
        # after k-pool compression is therefore the tight correctness bound.
        return (max_model_len + index_kpool - 1) // index_kpool

    # Generic DeepSeek behavior: keep the upstream multi-request workspace
    # heuristic unchanged for non-pooled indexers.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # (576 * 2 // 132) * 5 = 40.
    return max_model_len * 40
'''

text = TARGET.read_text()
if MARKER in text:
    raise SystemExit(0)
count = text.count(OLD)
if count != 1:
    raise RuntimeError(f"expected one indexer workspace anchor, found {count}")
TARGET.write_text(text.replace(OLD, NEW, 1))
