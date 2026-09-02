#!/usr/bin/env python3
"""Verify a full optimized-NVFP4 checkpoint against its Red Hat base."""

from __future__ import annotations

import argparse
import filecmp
import json
import math
import re
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


PACKED_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"weight_packed$"
)
OPTIMIZED_PATTERN = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\.(?:weight_packed|weight_scale)$"
)
OPTIMIZED_WITH_GLOBAL_PATTERN = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\."
    r"(?:weight_packed|weight_scale|weight_global_scale)$"
)


def load_index(root: Path) -> dict[str, str]:
    return json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]


def evenly_spaced(values: list[str], count: int) -> list[str]:
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(position * (len(values) - 1) / (count - 1))] for position in range(count)]


def companion_names(packed_name: str) -> tuple[str, str]:
    prefix = packed_name.removesuffix("weight_packed")
    return prefix + "weight_scale", prefix + "weight_global_scale"


def matrix_key(packed_name: str) -> tuple[int, int, str]:
    match = PACKED_PATTERN.match(packed_name)
    if match is None:
        raise ValueError(packed_name)
    return (
        int(match.group("layer")),
        int(match.group("expert")),
        match.group("projection"),
    )


def safetensors_payload_size(path: Path) -> int:
    """Return tensor-data bytes, excluding the variable-length JSON header."""
    with path.open("rb") as handle:
        encoded_header_size = handle.read(8)
    if len(encoded_header_size) != 8:
        raise AssertionError(f"truncated safetensors header: {path}")
    header_size = struct.unpack("<Q", encoded_header_size)[0]
    serialized_size = path.stat().st_size
    if header_size > serialized_size - 8:
        raise AssertionError(f"invalid safetensors header size: {path}")
    return serialized_size - 8 - header_size


def tensor_contract_and_samples(
    base_root: Path,
    candidate_root: Path,
    index: dict[str, str],
    samples_per_shard: int,
    *,
    allow_global_divisor_changes: bool,
) -> dict[str, Any]:
    names_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in index.items():
        names_by_shard[shard].append(name)
    result: dict[str, Any] = {}
    optimized_pattern = (
        OPTIMIZED_WITH_GLOBAL_PATTERN
        if allow_global_divisor_changes
        else OPTIMIZED_PATTERN
    )
    for shard, indexed_names in sorted(names_by_shard.items()):
        base_path = base_root / shard
        candidate_path = candidate_root / shard
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        base_payload_size = safetensors_payload_size(base_path)
        candidate_payload_size = safetensors_payload_size(candidate_path)
        if base_payload_size != candidate_payload_size:
            raise AssertionError(f"tensor payload size changed: {shard}")
        with safe_open(base_path, framework="pt", device="cpu") as base, safe_open(
            candidate_path, framework="pt", device="cpu"
        ) as candidate:
            base_names = list(base.keys())
            candidate_names = list(candidate.keys())
            if base_names != candidate_names or sorted(indexed_names) != base_names:
                raise AssertionError(f"tensor-key mismatch in {shard}")
            if base.metadata() != candidate.metadata():
                raise AssertionError(f"metadata mismatch in {shard}")
            for name in base_names:
                base_tensor = base.get_tensor(name)
                candidate_tensor = candidate.get_tensor(name)
                if base_tensor.shape != candidate_tensor.shape:
                    raise AssertionError(f"shape changed: {name}")
                if base_tensor.dtype != candidate_tensor.dtype:
                    raise AssertionError(f"dtype changed: {name}")

            packed_names = [name for name in base_names if PACKED_PATTERN.match(name)]
            changed_samples = evenly_spaced(packed_names, samples_per_shard)
            changed_packed_samples = 0
            changed_scale_samples = 0
            changed_global_samples = 0
            for packed_name in changed_samples:
                scale_name, global_name = companion_names(packed_name)
                packed_equal = torch.equal(
                    base.get_tensor(packed_name), candidate.get_tensor(packed_name)
                )
                scale_equal = torch.equal(
                    base.get_tensor(scale_name), candidate.get_tensor(scale_name)
                )
                if allow_global_divisor_changes:
                    changed_packed_samples += int(not packed_equal)
                    changed_scale_samples += int(not scale_equal)
                else:
                    if packed_equal:
                        raise AssertionError(
                            f"packed tensor did not change: {packed_name}"
                        )
                    if scale_equal:
                        raise AssertionError(
                            f"scale tensor did not change: {scale_name}"
                        )
                global_shard = index[global_name]
                if global_shard == shard:
                    global_equal = torch.equal(
                        base.get_tensor(global_name), candidate.get_tensor(global_name)
                    )
                else:
                    with safe_open(
                        base_root / global_shard, framework="pt", device="cpu"
                    ) as global_base, safe_open(
                        candidate_root / global_shard,
                        framework="pt",
                        device="cpu",
                    ) as global_candidate:
                        global_equal = torch.equal(
                            global_base.get_tensor(global_name),
                            global_candidate.get_tensor(global_name),
                        )
                if allow_global_divisor_changes:
                    changed_global_samples += int(not global_equal)
                elif not global_equal:
                    raise AssertionError(f"global divisor changed: {global_name}")

            unchanged_names = [
                name for name in base_names if not optimized_pattern.match(name)
            ]
            unchanged_samples = evenly_spaced(unchanged_names, samples_per_shard)
            for name in unchanged_samples:
                if not torch.equal(base.get_tensor(name), candidate.get_tensor(name)):
                    raise AssertionError(f"non-target tensor changed: {name}")
            result[shard] = {
                "bytes": candidate_path.stat().st_size,
                "payload_bytes": candidate_payload_size,
                "tensors": len(base_names),
                "sampled_target_matrices": len(changed_samples),
                "changed_packed_samples": changed_packed_samples,
                "changed_scale_samples": changed_scale_samples,
                "changed_global_divisor_samples": changed_global_samples,
                "unchanged_content_samples": len(unchanged_samples),
            }
    return result


def verify_reports(
    report_paths: list[Path],
    expected_matrices: set[tuple[int, int, str]],
) -> dict[str, Any]:
    reported: set[tuple[int, int, str]] = set()
    matrix_rows: dict[tuple[int, int, str], dict[str, Any]] = {}
    packed_rows: dict[tuple[int, int, str], dict[str, Any]] = {}
    rows = 0
    packed_report_rows = 0
    lower_boundary_max = 0.0
    upper_boundary_max = 0.0
    for path in report_paths:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                rows += 1
                key = (row["layer"], row["expert"], row["projection"])
                previous = matrix_rows.get(key)
                if previous is not None:
                    stable_fields = (
                        "shape",
                        "groups",
                        "values",
                        "original_global_divisor",
                        "selected_global_divisor",
                        "global_divisor_multiplier",
                        "global_divisor_search",
                        "search",
                    )
                    if any(previous[field] != row[field] for field in stable_fields):
                        raise AssertionError(f"inconsistent build-report rows for {key}")
                else:
                    matrix_rows[key] = row
                reported.add(key)
                if row.get("replace_packed"):
                    packed_report_rows += 1
                    packed_rows[key] = row
                lower_boundary_max = max(
                    lower_boundary_max, float(row["search"]["lower_boundary_fraction"])
                )
                upper_boundary_max = max(
                    upper_boundary_max, float(row["search"]["upper_boundary_fraction"])
                )
    if reported != expected_matrices:
        missing = sorted(expected_matrices - reported)[:20]
        extra = sorted(reported - expected_matrices)[:20]
        raise AssertionError(f"build-report coverage mismatch: missing={missing}, extra={extra}")
    result = {
        "report_rows": rows,
        "unique_matrices": len(reported),
        "duplicate_matrix_report_rows": rows - len(matrix_rows),
        "packed_report_rows": packed_report_rows,
        "duplicate_packed_report_rows": (
            packed_report_rows - len(packed_rows)
        ),
        "lower_boundary_fraction_max": lower_boundary_max,
        "upper_boundary_fraction_max": upper_boundary_max,
    }
    if set(packed_rows) != expected_matrices:
        missing = sorted(expected_matrices - set(packed_rows))[:20]
        extra = sorted(set(packed_rows) - expected_matrices)[:20]
        raise AssertionError(
            "packed build-report coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )

    gates = [
        row.get("global_divisor_search", {}).get("full_matrix_gate")
        for row in packed_rows.values()
    ]
    if all(gate is not None for gate in gates):
        baseline_sse = 0.0
        selected_pre_gate_sse = 0.0
        selected_sse = 0.0
        full_gate_fallbacks = 0
        heldout_fallbacks = 0
        changed_divisors = 0
        matrix_improvements: list[float] = []
        multiplier_histogram: Counter[str] = Counter()
        values = 0
        for row, gate in zip(packed_rows.values(), gates, strict=True):
            assert gate is not None
            count = int(row["values"])
            baseline_mse = float(gate["baseline_mse"])
            selected_pre_gate_mse = float(gate["selected_mse_before_gate"])
            full_fallback = bool(gate["fell_back"])
            selected_mse = baseline_mse if full_fallback else selected_pre_gate_mse
            multiplier = float(
                row["global_divisor_search"]["selected_multiplier"]
            )

            values += count
            baseline_sse += baseline_mse * count
            selected_pre_gate_sse += selected_pre_gate_mse * count
            selected_sse += selected_mse * count
            full_gate_fallbacks += int(full_fallback)
            heldout_fallbacks += int(
                bool(row["global_divisor_search"].get("fell_back"))
            )
            changed_divisors += int(multiplier != 1.0)
            multiplier_histogram[f"{multiplier:.12g}"] += 1
            matrix_improvements.append(1.0 - selected_mse / baseline_mse)

        mse_ratio = selected_sse / baseline_sse
        result["global_divisor_optimization"] = {
            "matrices": len(packed_rows),
            "values": values,
            "changed_divisors": changed_divisors,
            "heldout_gate_fallbacks": heldout_fallbacks,
            "full_matrix_gate_fallbacks": full_gate_fallbacks,
            "baseline_weighted_mse": baseline_sse / values,
            "selected_pre_gate_weighted_mse": selected_pre_gate_sse / values,
            "selected_weighted_mse": selected_sse / values,
            "weighted_mse_relative_reduction": 1.0 - mse_ratio,
            "weighted_rmse_relative_reduction": 1.0 - math.sqrt(mse_ratio),
            "per_matrix_mse_relative_reduction": {
                "minimum": min(matrix_improvements),
                "median": statistics.median(matrix_improvements),
                "mean": statistics.fmean(matrix_improvements),
                "maximum": max(matrix_improvements),
            },
            "selected_multiplier_histogram": dict(
                sorted(multiplier_histogram.items(), key=lambda item: float(item[0]))
            ),
        }
    return result


def verify_reference_replacements(
    candidate_root: Path,
    index: dict[str, str],
    reference_path: Path,
) -> dict[str, Any]:
    names_by_shard: dict[str, list[str]] = defaultdict(list)
    with safe_open(reference_path, framework="pt", device="cpu") as reference:
        reference_names = list(reference.keys())
        for name in reference_names:
            names_by_shard[index[name]].append(name)
        for shard, names in names_by_shard.items():
            with safe_open(
                candidate_root / shard, framework="pt", device="cpu"
            ) as candidate:
                for name in names:
                    if not torch.equal(reference.get_tensor(name), candidate.get_tensor(name)):
                        raise AssertionError(
                            f"full build differs from validated replacement: {name}"
                        )
    return {"reference_tensors": len(reference_names), "status": "exact_match"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, action="append", default=[])
    parser.add_argument("--reference-replacements", type=Path)
    parser.add_argument("--content-samples-per-shard", type=int, default=3)
    parser.add_argument("--allow-global-divisor-changes", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base_index = load_index(args.base_root)
    candidate_index = load_index(args.candidate_root)
    if base_index != candidate_index:
        raise AssertionError("model.safetensors.index.json weight_map changed")
    for filename in (
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    ):
        if not filecmp.cmp(
            args.base_root / filename,
            args.candidate_root / filename,
            shallow=False,
        ):
            raise AssertionError(f"serving metadata changed: {filename}")

    expected_matrices = {
        matrix_key(name) for name in base_index if PACKED_PATTERN.match(name)
    }
    report: dict[str, Any] = {
        "schema": 1,
        "base_root": str(args.base_root),
        "candidate_root": str(args.candidate_root),
        "expected_matrices": len(expected_matrices),
        "allow_global_divisor_changes": args.allow_global_divisor_changes,
        "tensor_contract": tensor_contract_and_samples(
            args.base_root,
            args.candidate_root,
            base_index,
            args.content_samples_per_shard,
            allow_global_divisor_changes=args.allow_global_divisor_changes,
        ),
        "expected_replacement_tensors": sum(
            bool(
                (
                    OPTIMIZED_WITH_GLOBAL_PATTERN
                    if args.allow_global_divisor_changes
                    else OPTIMIZED_PATTERN
                ).match(name)
            )
            for name in base_index
        ),
    }
    if args.build_report:
        report["build_reports"] = verify_reports(
            args.build_report, expected_matrices
        )
    if args.reference_replacements:
        report["reference_replacements"] = verify_reference_replacements(
            args.candidate_root,
            base_index,
            args.reference_replacements,
        )
    report["status"] = "PASS"
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
