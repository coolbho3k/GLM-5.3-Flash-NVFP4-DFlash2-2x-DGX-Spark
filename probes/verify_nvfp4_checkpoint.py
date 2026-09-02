#!/usr/bin/env python3
"""Verify a full optimized-NVFP4 checkpoint against its Red Hat base."""

from __future__ import annotations

import argparse
import filecmp
import json
import re
from collections import defaultdict
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
        if base_path.stat().st_size != candidate_path.stat().st_size:
            raise AssertionError(f"serialized shard size changed: {shard}")
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
    rows = 0
    lower_boundary_max = 0.0
    upper_boundary_max = 0.0
    for path in report_paths:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                rows += 1
                reported.add((row["layer"], row["expert"], row["projection"]))
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
    return {
        "report_rows": rows,
        "unique_matrices": len(reported),
        "lower_boundary_fraction_max": lower_boundary_max,
        "upper_boundary_fraction_max": upper_boundary_max,
    }


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
