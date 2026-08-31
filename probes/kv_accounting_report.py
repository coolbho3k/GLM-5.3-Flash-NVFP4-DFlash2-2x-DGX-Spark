#!/usr/bin/env python3
"""Summarize KV_ACCOUNTING records emitted by the instrumented vLLM image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MEMORY_MARKER = "KV_ACCOUNTING_MEMORY "
CONFIG_MARKER = "KV_ACCOUNTING_CONFIG "
DTYPE_MARKER = "KV_ACCOUNTING_DTYPES "


def records(lines: list[str], marker: str) -> list[dict[str, Any]]:
    found = []
    for line in lines:
        if marker in line:
            found.append(json.loads(line.split(marker, 1)[1]))
    return found


def gib(value: int) -> float:
    return round(value / (1024**3), 6)


def cdiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def capacity_matrix(
    limiting: dict[str, Any],
    request_lengths: list[int],
    dcp_size: int,
    max_in_flight_tokens: int,
) -> list[dict[str, Any]]:
    """Calculate per-request shared-pool demand from logged cache specs."""

    rows = []
    num_blocks = int(limiting["num_blocks"])
    pool_bytes_per_block = int(limiting["pool_bytes_per_block"])
    for request_tokens in request_lengths:
        components: dict[str, int] = {}
        slot_waste: dict[str, int] = {}
        for group in limiting["groups"]:
            specs = list(group.get("layer_specs", {}).values())
            if specs:
                types = {spec["type"] for spec in specs}
                spec = specs[0]
            else:
                spec = group["spec"]
                types = {spec["type"]}

            if "MLAAttentionSpec" in types:
                main_specs = [
                    value
                    for value in specs
                    if value["type"] == "MLAAttentionSpec"
                    and int(value.get("compress_ratio", 1)) == 1
                ]
                if not main_specs:
                    raise ValueError("MLA group has no uncompressed target spec")
                block_size = int(main_specs[0]["block_size"])
                blocks = cdiv(request_tokens, block_size * dcp_size)
                category = "target_mla_and_indexer"
                slot_waste[category] = (
                    blocks * block_size * dcp_size - request_tokens
                )
            elif "KpoolTailSpec" in types:
                blocks = cdiv(
                    int(spec["max_request_bytes"]), int(spec["page_size_bytes"])
                )
                category = "indexer_tail"
            elif "SlidingWindowSpec" in types:
                block_size = int(spec["block_size"])
                held_tokens = min(
                    int(spec["sliding_window"]) - 1 + max_in_flight_tokens,
                    request_tokens,
                )
                # One extra block is required because the circular window can
                # begin partway through a physical page.
                blocks = cdiv(held_tokens, block_size) + 1
                category = "dflash_window"
                slot_waste[category] = blocks * block_size - held_tokens
            elif "MambaSpec" in types:
                blocks = cdiv(
                    int(spec["max_request_bytes"]), int(spec["page_size_bytes"])
                )
                category = "kda_recurrent"
            else:
                blocks = cdiv(
                    int(spec["max_request_bytes"]), int(spec["page_size_bytes"])
                )
                category = "+".join(sorted(types))
            components[category] = components.get(category, 0) + blocks

        blocks_per_request = sum(components.values())
        fractional_concurrency = num_blocks / blocks_per_request
        rows.append(
            {
                "request_tokens": request_tokens,
                "pool_blocks_per_request": blocks_per_request,
                "components_blocks": components,
                "slot_waste_tokens": slot_waste,
                "whole_concurrent_requests": num_blocks // blocks_per_request,
                "fractional_concurrency": round(fractional_concurrency, 6),
                "effective_logical_token_capacity": int(
                    fractional_concurrency * request_tokens
                ),
                "physical_reservation_gib_per_request": gib(
                    blocks_per_request * pool_bytes_per_block
                ),
            }
        )
    return rows


def summarize(
    lines: list[str],
    *,
    request_lengths: list[int] | None = None,
    dcp_size: int = 2,
    max_in_flight_tokens: int = 16384,
) -> dict[str, Any]:
    memory = records(lines, MEMORY_MARKER)
    configs = records(lines, CONFIG_MARKER)
    dtype_inventories = records(lines, DTYPE_MARKER)
    if not memory:
        raise ValueError("no KV_ACCOUNTING_MEMORY records found")
    if not configs:
        raise ValueError("no KV_ACCOUNTING_CONFIG records found")

    for item in memory:
        if item["requested_identity_error_bytes"] != 0:
            raise ValueError(f"rank {item['rank']} memory identity does not close")
    for item in configs:
        if item["tensor_identity_error_bytes"] != 0:
            raise ValueError("cache tensor identity does not close")
        if item["allocator_tail_bytes"] < 0:
            raise ValueError("cache tensors exceed available memory")

    limiting = min(configs, key=lambda item: item["num_blocks"])
    num_blocks = limiting["num_blocks"]
    components = {"mla_shared": 0, "dsa_indexer_and_tail": 0, "other": 0}
    for tensor in limiting["tensors"]:
        owner = tensor["shared_by"][0]
        if "indexer.k_cache" in owner:
            category = "dsa_indexer_and_tail"
        elif ".attn" in owner:
            category = "mla_shared"
        else:
            category = "other"
        components[category] += tensor["size_bytes"] // num_blocks

    manager_block_size = 0
    for group in limiting["groups"]:
        specs = group.get("layer_specs", {})
        for spec in specs.values():
            manager_block_size = max(manager_block_size, int(spec.get("block_size", 0)))

    memory_summary = []
    for item in sorted(memory, key=lambda value: value["rank"]):
        persistent_other = item["total_consumed_bytes"] - item["weights_bytes"]
        memory_summary.append(
            {
                "rank": item["rank"],
                "requested_gib": gib(item["requested_bytes"]),
                "weights_gib": gib(item["weights_bytes"]),
                "persistent_non_weight_gib": gib(persistent_other),
                "transient_peak_gib": gib(item["transient_peak_headroom_bytes"]),
                "non_kv_total_gib": gib(item["non_kv_cache_bytes"]),
                "available_kv_gib": gib(item["available_kv_bytes"]),
                "identity_error_bytes": item["requested_identity_error_bytes"],
            }
        )

    matrix = capacity_matrix(
        limiting,
        request_lengths or [2048, 8192, 32768, 65536, 131072, 262144],
        dcp_size,
        max_in_flight_tokens,
    )
    result = {
        "schema": 1,
        "memory_ranks": memory_summary,
        "cache_candidates": [
            {
                "available_gib": gib(item["available_bytes"]),
                "num_blocks": item["num_blocks"],
                "pool_bytes_per_block": item["pool_bytes_per_block"],
                "tensor_gib": gib(item["tensor_bytes"]),
                "allocator_tail_bytes": item["allocator_tail_bytes"],
                "identity_error_bytes": item["tensor_identity_error_bytes"],
            }
            for item in configs
        ],
        "limiting_cache": {
            "num_blocks": num_blocks,
            "manager_block_tokens": manager_block_size,
            "pool_bytes_per_block": limiting["pool_bytes_per_block"],
            "physical_bytes_per_manager_token": round(
                limiting["pool_bytes_per_block"] / manager_block_size, 6
            ),
            "components_bytes_per_block": components,
            "allocator_tail_bytes": limiting["allocator_tail_bytes"],
        },
        "capacity_matrix": matrix,
        "mixed_length_portfolio": {
            "request_lengths": [row["request_tokens"] for row in matrix],
            "tokens_per_portfolio": sum(row["request_tokens"] for row in matrix),
            "pool_blocks_per_portfolio": sum(
                row["pool_blocks_per_request"] for row in matrix
            ),
            "whole_portfolios": num_blocks
            // sum(row["pool_blocks_per_request"] for row in matrix),
            "fractional_portfolios": round(
                num_blocks
                / sum(row["pool_blocks_per_request"] for row in matrix),
                6,
            ),
            "effective_logical_token_capacity": int(
                num_blocks
                / sum(row["pool_blocks_per_request"] for row in matrix)
                * sum(row["request_tokens"] for row in matrix)
            ),
        },
    }
    if dtype_inventories:
        dtype_inventories = sorted(
            dtype_inventories, key=lambda item: item["rank"]
        )
        result["resident_dtype_inventories"] = dtype_inventories

        # Upper bounds only: real weight/buffer quantization needs scales,
        # kernel support, and a fresh quality/performance profile. This makes
        # the opportunity explicit without claiming those bytes are free.
        bf16_bytes = min(
            int(item["combined_by_dtype"].get("bfloat16", 0))
            for item in dtype_inventories
        )
        longest = max(matrix, key=lambda row: row["request_tokens"])
        blocks_per_request = int(longest["pool_blocks_per_request"])
        projections = []
        for target_bits in (8, 4):
            saved_bytes = bf16_bytes * (16 - target_bits) // 16
            extra_blocks = saved_bytes // int(limiting["pool_bytes_per_block"])
            projected_blocks = num_blocks + extra_blocks
            projections.append(
                {
                    "target_bits": target_bits,
                    "source_bf16_bytes": bf16_bytes,
                    "ideal_saved_bytes": saved_bytes,
                    "ideal_extra_pool_blocks": extra_blocks,
                    "ideal_projected_pool_blocks": projected_blocks,
                    "ideal_projected_longest_request_token_capacity": int(
                        projected_blocks
                        / blocks_per_request
                        * int(longest["request_tokens"])
                    ),
                }
            )
        result["bf16_quantization_upper_bounds"] = {
            "assumption": (
                "all inventoried BF16 storage converts with no scale/metadata, "
                "activation, quality, or serving-performance cost"
            ),
            "projections": projections,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--request-lengths",
        default="2048,8192,32768,65536,131072,262144",
        help="comma-separated request lengths for the concurrency matrix",
    )
    parser.add_argument("--dcp-size", type=int, default=2)
    parser.add_argument("--max-in-flight-tokens", type=int, default=16384)
    args = parser.parse_args()
    if args.logs:
        lines = []
        for path in args.logs:
            lines.extend(path.read_text(errors="replace").splitlines())
    else:
        lines = sys.stdin.read().splitlines()
    request_lengths = [
        int(value) for value in args.request_lengths.split(",") if value.strip()
    ]
    result = summarize(
        lines,
        request_lengths=request_lengths,
        dcp_size=args.dcp_size,
        max_in_flight_tokens=args.max_in_flight_tokens,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
