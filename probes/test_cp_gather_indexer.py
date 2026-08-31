#!/usr/bin/env python3
"""Byte-exact checks for vLLM's paged indexer-cache gather operator."""

from __future__ import annotations

import argparse
import json

import torch


def expected_gather(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    block_size = cache.shape[1]
    for request, seq_len in enumerate(seq_lens):
        for position in range(seq_len):
            block = int(block_table[request, position // block_size])
            rows.append(cache[block, position % block_size])
    return torch.stack(rows)


def run_case(record_bytes: int, *, blocks: int = 7, block_size: int = 4) -> dict:
    value_bytes = record_bytes - 4
    records = torch.arange(
        blocks * block_size * record_bytes, dtype=torch.int64
    ).remainder(251).to(torch.uint8).reshape(blocks, block_size, record_bytes)
    # vLLM's cache tensor is shaped [blocks, block_size, record_bytes], but its
    # physical page is split: all values first, then all scales. Populate that
    # byte layout while retaining the public tensor geometry.
    host = torch.empty_like(records)
    pages = host.view(blocks, -1)
    pages[:, : block_size * value_bytes] = records[:, :, :value_bytes].reshape(
        blocks, -1
    )
    pages[:, block_size * value_bytes :] = records[:, :, value_bytes:].reshape(
        blocks, -1
    )
    block_table_host = torch.tensor([[3, 1, 4], [2, 0, 5]], dtype=torch.int32)
    seq_lens = [6, 9]
    cu_seq_lens = torch.tensor([0, seq_lens[0], sum(seq_lens)], dtype=torch.int32)

    cache = host.cuda()
    block_table = block_table_host.cuda()
    dst_k = torch.full(
        (sum(seq_lens), value_bytes), 0xFE, dtype=torch.uint8, device="cuda"
    )
    dst_scale = torch.full(
        (sum(seq_lens), 4), 0xFD, dtype=torch.uint8, device="cuda"
    )
    torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
        cache, dst_k, dst_scale, block_table, cu_seq_lens.cuda()
    )
    torch.cuda.synchronize()

    expected = expected_gather(records, block_table_host, seq_lens)
    got = torch.cat((dst_k.cpu(), dst_scale.cpu()), dim=1)
    mismatches = int((got != expected).sum())
    return {
        "record_bytes": record_bytes,
        "rows": sum(seq_lens),
        "mismatched_bytes": mismatches,
        "ok": mismatches == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--library",
        default="/usr/local/lib/python3.12/dist-packages/vllm/"
        "_C_stable_libtorch.abi3.so",
    )
    args = parser.parse_args()
    torch.ops.load_library(args.library)

    results = [run_case(132), run_case(68)]
    report = {"ok": all(result["ok"] for result in results), "results": results}
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
