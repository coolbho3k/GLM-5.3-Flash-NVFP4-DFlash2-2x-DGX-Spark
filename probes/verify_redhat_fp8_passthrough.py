#!/usr/bin/env python3
"""Verify the Red Hat FP8-passthrough checkpoint repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from repair_redhat_fp8_passthrough import (
    COPY_BUFFER_BYTES,
    discover_restore_weights,
    load_index,
    locate_shard,
    output_scale_name,
    read_safetensors_header,
    restore_family,
    source_scale_name,
)


def tensor_digest(path: Path, entry: dict[str, Any], data_start: int) -> str:
    start, end = (int(value) for value in entry["data_offsets"])
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(data_start + start)
        remaining = end - start
        while remaining:
            chunk = handle.read(min(remaining, COPY_BUFFER_BYTES))
            if not chunk:
                raise EOFError(f"short tensor read in {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def require_equal_tensor_bytes(
    left_path: Path,
    left_entry: dict[str, Any],
    left_start: int,
    right_path: Path,
    right_entry: dict[str, Any],
    right_start: int,
    name: str,
) -> None:
    left_size = int(left_entry["data_offsets"][1]) - int(
        left_entry["data_offsets"][0]
    )
    right_size = int(right_entry["data_offsets"][1]) - int(
        right_entry["data_offsets"][0]
    )
    if left_size != right_size:
        raise AssertionError(f"serialized size differs for {name}")
    if tensor_digest(left_path, left_entry, left_start) != tensor_digest(
        right_path, right_entry, right_start
    ):
        raise AssertionError(f"serialized bytes differ for {name}")


def numeric_equivalence(
    base_root: Path,
    source_roots: list[Path],
    base_index: dict[str, str],
    source_index: dict[str, str],
    restore_weights: list[str],
    row_chunk: int,
) -> dict[str, Any]:
    checked_values = 0
    max_abs_error = 0.0
    for position, name in enumerate(restore_weights, 1):
        base_path = base_root / base_index[name]
        source_path = locate_shard(source_index[name], source_roots)
        scale_name = source_scale_name(name)
        scale_path = locate_shard(source_index[scale_name], source_roots)
        with safe_open(base_path, framework="pt", device="cpu") as base, safe_open(
            source_path, framework="pt", device="cpu"
        ) as source_weight, safe_open(
            scale_path, framework="pt", device="cpu"
        ) as source_scale:
            bf16_slice = base.get_slice(name)
            fp8_slice = source_weight.get_slice(name)
            scale = source_scale.get_tensor(scale_name).float()
            rows, columns = bf16_slice.get_shape()
            block_rows = math.ceil(rows / 128)
            block_columns = math.ceil(columns / 128)
            if list(scale.shape) != [block_rows, block_columns]:
                raise AssertionError(f"scale shape mismatch for {name}")
            for row_start in range(0, rows, row_chunk):
                row_end = min(row_start + row_chunk, rows)
                fp8 = fp8_slice[row_start:row_end].float()
                bf16 = bf16_slice[row_start:row_end]
                scale_rows = scale[row_start // 128 : math.ceil(row_end / 128)]
                expanded = scale_rows.repeat_interleave(128, 0).repeat_interleave(
                    128, 1
                )
                row_offset = row_start % 128
                expanded = expanded[row_offset : row_offset + row_end - row_start]
                reconstructed = (fp8 * expanded[:, :columns]).to(torch.bfloat16)
                if not torch.equal(reconstructed, bf16):
                    error = (reconstructed.float() - bf16.float()).abs().max().item()
                    max_abs_error = max(max_abs_error, error)
                    raise AssertionError(
                        f"Red Hat BF16 differs from official FP8 dequantization: "
                        f"{name}, max_abs_error={error}"
                    )
                checked_values += bf16.numel()
        print(f"numeric {position}/{len(restore_weights)} {name}", flush=True)
    return {
        "weights": len(restore_weights),
        "values": checked_values,
        "max_abs_error": max_abs_error,
        "status": "exact_bf16_match",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--skip-full-content", action="store_true")
    parser.add_argument("--numeric-equivalence", action="store_true")
    parser.add_argument("--numeric-row-chunk", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base_root = args.base_root.resolve()
    source_roots = [path.resolve() for path in args.source_root]
    candidate_root = args.candidate_root.resolve()
    base_index_doc = load_index(base_root)
    source_index_doc = load_index(source_roots[0])
    candidate_index_doc = load_index(candidate_root)
    base_index = base_index_doc["weight_map"]
    source_index = source_index_doc["weight_map"]
    candidate_index = candidate_index_doc["weight_map"]
    restore_weights = discover_restore_weights(source_index, base_index)
    restore_scales = {output_scale_name(name) for name in restore_weights}

    expected_names = set(base_index) | restore_scales
    if set(candidate_index) != expected_names:
        missing = sorted(expected_names - set(candidate_index))[:20]
        extra = sorted(set(candidate_index) - expected_names)[:20]
        raise AssertionError(f"candidate index mismatch: missing={missing}, extra={extra}")

    config = json.loads((candidate_root / "config.json").read_text())
    quant = config["quantization_config"]
    if quant.get("format") != "mixed-precision":
        raise AssertionError("candidate is not marked mixed-precision")
    if quant["config_groups"]["group_1"].get("input_activations") is not None:
        raise AssertionError("FP8 passthrough must preserve BF16 activations")

    headers: dict[Path, tuple[dict[str, Any], int]] = {}

    def header(path: Path) -> tuple[dict[str, Any], int]:
        if path not in headers:
            headers[path] = read_safetensors_header(path)
        return headers[path]

    bytes_checked = 0
    unchanged_checked = 0
    restored_checked = 0
    for name, base_shard in sorted(base_index.items()):
        base_path = base_root / base_shard
        candidate_path = candidate_root / candidate_index[name]
        base_header, base_start = header(base_path)
        candidate_header, candidate_start = header(candidate_path)
        if name in restore_weights:
            source_path = locate_shard(source_index[name], source_roots)
            source_header, source_start = header(source_path)
            expected = source_header[name]
            if candidate_header[name]["dtype"] != "F8_E4M3":
                raise AssertionError(f"restored weight is not FP8: {name}")
            if candidate_header[name]["shape"] != expected["shape"]:
                raise AssertionError(f"restored weight shape mismatch: {name}")
            if not args.skip_full_content:
                require_equal_tensor_bytes(
                    candidate_path,
                    candidate_header[name],
                    candidate_start,
                    source_path,
                    expected,
                    source_start,
                    name,
                )
            restored_checked += 1
        else:
            if candidate_header[name]["dtype"] != base_header[name]["dtype"]:
                raise AssertionError(f"unchanged dtype differs: {name}")
            if candidate_header[name]["shape"] != base_header[name]["shape"]:
                raise AssertionError(f"unchanged shape differs: {name}")
            if not args.skip_full_content:
                require_equal_tensor_bytes(
                    candidate_path,
                    candidate_header[name],
                    candidate_start,
                    base_path,
                    base_header[name],
                    base_start,
                    name,
                )
            unchanged_checked += 1
        entry = candidate_header[name]
        bytes_checked += int(entry["data_offsets"][1]) - int(entry["data_offsets"][0])

    for weight_name in restore_weights:
        candidate_name = output_scale_name(weight_name)
        candidate_path = candidate_root / candidate_index[candidate_name]
        candidate_header, candidate_start = header(candidate_path)
        source_name = source_scale_name(weight_name)
        source_path = locate_shard(source_index[source_name], source_roots)
        source_header, source_start = header(source_path)
        if candidate_header[candidate_name]["dtype"] != "F32":
            raise AssertionError(f"restored scale is not F32: {candidate_name}")
        if not args.skip_full_content:
            require_equal_tensor_bytes(
                candidate_path,
                candidate_header[candidate_name],
                candidate_start,
                source_path,
                source_header[source_name],
                source_start,
                candidate_name,
            )

    report: dict[str, Any] = {
        "schema": 1,
        "restored_weights": restored_checked,
        "restored_scales": len(restore_scales),
        "unchanged_tensors": unchanged_checked,
        "family_counts": dict(
            Counter(restore_family(name) for name in restore_weights)
        ),
        "full_content_verified": not args.skip_full_content,
        "candidate_tensor_bytes": int(
            candidate_index_doc.get("metadata", {}).get("total_size", bytes_checked)
        ),
    }
    if args.numeric_equivalence:
        report["numeric_equivalence"] = numeric_equivalence(
            base_root,
            source_roots,
            base_index,
            source_index,
            restore_weights,
            args.numeric_row_chunk,
        )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
