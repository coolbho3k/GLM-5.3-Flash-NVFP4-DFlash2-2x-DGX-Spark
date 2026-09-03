#!/usr/bin/env python3
"""Isolated GPU gates for the GLM pooled DCP2 + MXFP4 indexer path."""

import json

import torch

from vllm import _custom_ops as ops
from b12x.attention.dsa_indexer.sm121_mxfp4 import (
    fwht128_quant_mxfp4,
    get_compressed_pool_slot_mapping_dcp,
    kpool_compress_and_write_cache_mxfp4,
    mqa_logits_mxfp4,
    paged_mqa_logits_mxfp4,
)
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits


def check_pool_slots() -> dict[str, list[int]]:
    query_start = torch.tensor([0, 12, 20], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([12, 8], dtype=torch.int32, device="cuda")
    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32, device="cuda")
    expected = {
        0: {3: 40, 11: 41, 15: 80},
        1: {7: 40, 19: 80},
    }
    results: dict[str, list[int]] = {}
    for rank in (0, 1):
        storage = torch.empty(20, dtype=torch.int64, device="cuda")
        actual = get_compressed_pool_slot_mapping_dcp(
            20,
            query_start,
            seq_lens,
            block_table,
            storage_block_size=4,
            compress_ratio=4,
            dcp_world_size=2,
            dcp_rank=rank,
            out=storage,
        ).cpu()
        reference = torch.full((20,), -1, dtype=torch.int64)
        for index, slot in expected[rank].items():
            reference[index] = slot
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        results[f"rank{rank}"] = actual.tolist()
    return results


def check_scorers() -> dict[str, float | int]:
    torch.manual_seed(20260830)
    rows, heads, keys, dim = 8, 32, 1024, 128
    q = torch.randn(rows * heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(keys, dim, device="cuda", dtype=torch.bfloat16)
    q_values, q_scales = fwht128_quant_mxfp4(q)
    k_values, k_scales = fwht128_quant_mxfp4(k)
    q_values = q_values.view(rows, heads, dim // 2)
    q_scales = q_scales.view(rows, heads, dim // 32)
    q_scales_packed = q_scales.view(torch.int32).squeeze(-1)
    weights = torch.rand(rows, heads, device="cuda", dtype=torch.float32)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.full((rows,), keys, device="cuda", dtype=torch.int32)

    reference = mqa_logits_mxfp4(
        (q_values, q_scales_packed),
        (k_values, k_scales),
        weights,
        starts,
        ends,
    )
    native = fp8_fp4_mqa_logits(
        (q_values.view(torch.int8), q_scales_packed),
        (k_values.view(torch.int8), k_scales.view(torch.int32).squeeze(-1)),
        weights,
        starts,
        ends,
        clean_logits=False,
    )
    torch.cuda.synchronize()
    difference = (reference - native).abs()
    selected = 512
    ref_topk = torch.topk(reference, selected, dim=1).indices
    native_topk = torch.topk(native, selected, dim=1).indices
    recalls = [
        torch.isin(ref_topk[row], native_topk[row]).float().mean().item()
        for row in range(rows)
    ]
    return {
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "top512_recall_min": min(recalls),
        "top512_recall_mean": sum(recalls) / len(recalls),
        "exact_top512_rows": sum(recall == 1.0 for recall in recalls),
        "rows": rows,
    }


def check_writer_gather() -> dict[str, float | int | bool]:
    """Verify the writer, vLLM gather, and paged reader share one layout."""
    torch.manual_seed(20260831)
    rows, pool_size, dim = 6, 4, 128
    blocks, block_size = 3, 4
    keys = torch.randn(
        rows, pool_size, dim, device="cuda", dtype=torch.bfloat16
    )
    scores = torch.randn_like(keys)
    ape = torch.randn(pool_size, dim, device="cuda", dtype=torch.float32)
    block_table = torch.tensor([[2, 0]], dtype=torch.int32, device="cuda")
    locations = torch.tensor([8, 9, 10, 11, 0, 1], device="cuda", dtype=torch.int64)
    cache = torch.full(
        (blocks, block_size, 68), 0xCD, device="cuda", dtype=torch.uint8
    )

    kpool_compress_and_write_cache_mxfp4(
        cache, keys, scores, ape, locations
    )
    gathered_values = torch.empty((rows, 64), device="cuda", dtype=torch.uint8)
    gathered_scales = torch.empty((rows, 4), device="cuda", dtype=torch.uint8)
    cu_seq_lens = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    ops.cp_gather_indexer_k_quant_cache(
        cache,
        gathered_values,
        gathered_scales,
        block_table,
        cu_seq_lens,
    )

    probabilities = torch.softmax(scores.float() + ape[None], dim=1)
    pooled = (keys.float() * probabilities).sum(dim=1).to(torch.bfloat16)
    expected_values, expected_scales = fwht128_quant_mxfp4(pooled)
    values_exact = torch.equal(gathered_values, expected_values)
    scales_exact = torch.equal(gathered_scales, expected_scales)

    heads = 32
    query = torch.randn(heads, dim, device="cuda", dtype=torch.bfloat16)
    q_values, q_scales = fwht128_quant_mxfp4(query)
    q_values = q_values.view(1, 1, heads, 64)
    q_scales_packed = q_scales.view(1, 1, heads, 4).view(torch.int32)
    weights = torch.rand(1, heads, device="cuda", dtype=torch.float32)
    starts = torch.zeros(1, device="cuda", dtype=torch.int32)
    ends = torch.full((1,), rows, device="cuda", dtype=torch.int32)
    contiguous = mqa_logits_mxfp4(
        (q_values.view(1, heads, 64), q_scales_packed.view(1, heads)),
        (gathered_values, gathered_scales),
        weights,
        starts,
        ends,
    )
    paged = paged_mqa_logits_mxfp4(
        (q_values, q_scales_packed),
        cache.unsqueeze(2),
        weights,
        torch.tensor([rows], dtype=torch.int32, device="cuda"),
        block_table,
        max_model_len=rows,
    )
    torch.cuda.synchronize()
    difference = (contiguous - paged[:, :rows]).abs()
    return {
        "values_exact": values_exact,
        "scales_exact": scales_exact,
        "paged_max_abs": difference.max().item(),
        "paged_mean_abs": difference.mean().item(),
        "ok": values_exact and scales_exact and difference.max().item() < 2e-5,
    }


def check_native_k7_paged() -> dict[str, float | int | bool]:
    """Verify one K=7 target step keeps all eight causal rows correct."""
    torch.manual_seed(20260901)
    batch, next_n, heads, dim = 1, 8, 32, 128
    keys, page_size = 16, 4
    key = torch.randn(keys, dim, device="cuda", dtype=torch.bfloat16)
    key_values, key_scales = fwht128_quant_mxfp4(key)
    raw_cache = torch.empty(
        (keys // page_size, page_size, 68), dtype=torch.uint8, device="cuda"
    )
    # Populate the physical per-page [all values][all scales] layout through
    # the production writer. The 4-D tensor below is only DeepGEMM's logical
    # view; assigning its apparent 68-byte records directly would be wrong.
    kpool_compress_and_write_cache_mxfp4(
        raw_cache,
        key[:, None, :],
        torch.zeros(keys, 1, dim, dtype=torch.bfloat16, device="cuda"),
        torch.zeros(1, dim, dtype=torch.float32, device="cuda"),
        torch.arange(keys, dtype=torch.int64, device="cuda"),
    )
    cache = raw_cache.unsqueeze(2)

    query = torch.randn(
        batch * next_n * heads, dim, device="cuda", dtype=torch.bfloat16
    )
    q_values, q_scales = fwht128_quant_mxfp4(query)
    q_values = q_values.view(batch, next_n, heads, 64)
    q_scales = q_scales.view(batch, next_n, heads, 4).view(torch.int32)
    weights = torch.rand(
        batch * next_n, heads, device="cuda", dtype=torch.float32
    )
    context_lens = torch.arange(
        keys - next_n + 1, keys + 1, dtype=torch.int32, device="cuda"
    ).view(batch, next_n)
    block_table = torch.arange(
        keys // page_size, dtype=torch.int32, device="cuda"
    ).view(batch, -1)

    paged = paged_mqa_logits_mxfp4(
        (q_values, q_scales),
        cache,
        weights,
        context_lens,
        block_table,
        max_model_len=keys,
    )
    contiguous = mqa_logits_mxfp4(
        (q_values.view(-1, heads, 64), q_scales.view(-1, heads)),
        (key_values, key_scales),
        weights,
        torch.zeros(batch * next_n, dtype=torch.int32, device="cuda"),
        context_lens.reshape(-1),
    )
    torch.cuda.synchronize()
    live = torch.arange(keys, device="cuda")[None, :] < context_lens.reshape(-1, 1)
    difference = (paged[live] - contiguous[live]).abs()
    dead_are_inf = bool(torch.isneginf(paged[~live]).all().item())
    max_abs = difference.max().item()
    return {
        "rows": batch * next_n,
        "max_abs": max_abs,
        "mean_abs": difference.mean().item(),
        "dead_are_inf": dead_are_inf,
        "ok": dead_are_inf and max_abs < 2e-5,
    }


def main() -> None:
    print(
        json.dumps(
            {
                "pool_slots": check_pool_slots(),
                "scorers": check_scorers(),
                "writer_gather": check_writer_gather(),
                "native_k7_paged": check_native_k7_paged(),
            }
        )
    )


if __name__ == "__main__":
    main()
