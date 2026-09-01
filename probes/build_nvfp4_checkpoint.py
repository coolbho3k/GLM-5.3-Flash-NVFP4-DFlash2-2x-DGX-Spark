#!/usr/bin/env python3
"""Build a serving-compatible NVFP4 checkpoint with optimized group scales.

This is the streaming/full-checkpoint companion to
``optimize_nvfp4_rounding.py``.  It rewrites selected Red Hat safetensor
shards one at a time, replacing only routed-expert ``weight_packed`` and
``weight_scale`` tensors.  Tensor names, shapes, dtypes, global divisors,
configuration, and the serving kernel contract remain unchanged.

The output directory should first be a copy-on-write clone of the Red Hat
checkpoint.  Shard-level state makes interrupted builds resumable and allows
different target shards to be built independently on the two cluster nodes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from optimize_nvfp4_rounding import (
    dequant_source,
    load_index,
    load_tensor,
    optimize_groups,
    source_names,
    target_names,
)


PACKED_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"weight_packed$"
)


def parse_csv(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated item")
    return result


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def shard_number(name: str) -> int:
    match = re.search(r"model-(\d+)-of-\d+\.safetensors$", name)
    return int(match.group(1)) if match else 1 << 30


def packed_plan(
    source_index: dict[str, str],
    target_index: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for packed_name, target_shard in target_index.items():
        match = PACKED_PATTERN.match(packed_name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        projection = match.group("projection")
        source_weight, source_scale = source_names(layer, expert, projection)
        _, target_scale, global_scale = target_names(layer, expert, projection)
        missing = [
            name
            for name in (source_weight, source_scale)
            if name not in source_index
        ]
        if missing:
            raise KeyError(f"official source index is missing {missing}")
        if target_scale not in target_index or global_scale not in target_index:
            raise KeyError(f"target companions are missing for {packed_name}")
        entry = {
                "layer": layer,
                "expert": expert,
                "projection": projection,
                "packed_name": packed_name,
                "target_scale_name": target_scale,
                "global_scale_name": global_scale,
                "source_weight_name": source_weight,
                "source_scale_name": source_scale,
                "source_weight_shard": source_index[source_weight],
                "source_scale_shard": source_index[source_scale],
        }
        packed_shard = target_index[packed_name]
        scale_shard = target_index[target_scale]
        for destination_shard in {packed_shard, scale_shard}:
            by_shard[destination_shard].append(
                {
                    **entry,
                    "replace_packed": destination_shard == packed_shard,
                    "replace_scale": destination_shard == scale_shard,
                }
            )
    for entries in by_shard.values():
        entries.sort(key=lambda row: (row["layer"], row["expert"], row["projection"]))
    return dict(sorted(by_shard.items(), key=lambda item: shard_number(item[0])))


def plan_summary(plan: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"target_shards": {}}
    all_source_shards: set[str] = set()
    total = 0
    unique_matrices: set[tuple[int, int, str]] = set()
    for target_shard, entries in plan.items():
        source_shards = sorted(
            {
                row[key]
                for row in entries
                for key in ("source_weight_shard", "source_scale_shard")
            },
            key=shard_number,
        )
        all_source_shards.update(source_shards)
        total += len(entries)
        unique_matrices.update(
            (row["layer"], row["expert"], row["projection"]) for row in entries
        )
        result["target_shards"][target_shard] = {
            "matrix_computations": len(entries),
            "packed_replacements": sum(bool(row["replace_packed"]) for row in entries),
            "scale_replacements": sum(bool(row["replace_scale"]) for row in entries),
            "layers": sorted({row["layer"] for row in entries}),
            "source_shards": source_shards,
        }
    result["matrix_computations"] = total
    result["unique_matrices"] = len(unique_matrices)
    result["source_shards"] = sorted(all_source_shards, key=shard_number)
    return result


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema": 1, "completed_shards": {}, "started_at": time.time()}
    value = json.loads(path.read_text())
    if value.get("schema") != 1:
        raise ValueError(f"unsupported state schema in {path}")
    value.setdefault("completed_shards", {})
    return value


def summarize_search(rows: list[dict[str, Any]]) -> dict[str, Any]:
    histogram: Counter[str] = Counter()
    changed_numerator = 0.0
    boundary_low = 0.0
    boundary_high = 0.0
    clipped = 0.0
    for row in rows:
        search = row["search"]
        groups = int(row["groups"])
        changed_numerator += float(search["scale_code_changed_fraction"]) * groups
        boundary_low += float(search["lower_boundary_fraction"]) * groups
        boundary_high += float(search["upper_boundary_fraction"]) * groups
        clipped += float(search["clipped_value_fraction"]) * int(row["values"])
        for delta, count in search["scale_code_delta_histogram"].items():
            histogram[delta] += int(count)
    total_groups = sum(int(row["groups"]) for row in rows)
    total_values = sum(int(row["values"]) for row in rows)
    return {
        "matrices": len(rows),
        "groups": total_groups,
        "values": total_values,
        "scale_code_changed_fraction": changed_numerator / total_groups,
        "lower_boundary_fraction": boundary_low / total_groups,
        "upper_boundary_fraction": boundary_high / total_groups,
        "clipped_value_fraction": clipped / total_values,
        "scale_code_delta_histogram": dict(
            sorted(histogram.items(), key=lambda item: int(item[0]))
        ),
    }


@torch.no_grad()
def build_shard(
    *,
    source_root: Path,
    source_index: dict[str, str],
    target_root: Path,
    target_index: dict[str, str],
    output_root: Path,
    target_shard: str,
    entries: list[dict[str, Any]],
    device: torch.device,
    scale_radius_below: int,
    scale_radius_above: int,
    row_chunk: int,
    report_handle,
) -> dict[str, Any]:
    base_path = target_root / target_shard
    destination = output_root / target_shard
    temporary = destination.with_name(f".{destination.name}.nvfp4-build.tmp")
    if not base_path.is_file():
        raise FileNotFoundError(base_path)

    with safe_open(base_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}

    shard_started = time.time()
    rows: list[dict[str, Any]] = []
    for position, entry in enumerate(entries, 1):
        matrix_started = time.time()
        source_weight = load_tensor(
            source_root,
            source_index,
            entry["source_weight_name"],
            device=device,
        )
        source_scale = load_tensor(
            source_root,
            source_index,
            entry["source_scale_name"],
            device=device,
        )
        reference = dequant_source(source_weight, source_scale)
        if entry["global_scale_name"] in tensors:
            global_divisor = tensors[entry["global_scale_name"]].to(device)
        else:
            global_divisor = load_tensor(
                target_root,
                target_index,
                entry["global_scale_name"],
                device=device,
            )
        candidate_packed, candidate_scale, search = optimize_groups(
            reference,
            global_divisor,
            scale_radius_below=scale_radius_below,
            scale_radius_above=scale_radius_above,
            row_chunk=row_chunk,
            input_second_moment=None,
        )
        if entry["replace_packed"]:
            if tuple(candidate_packed.shape) != tuple(
                tensors[entry["packed_name"]].shape
            ):
                raise ValueError(f"shape changed for {entry['packed_name']}")
            tensors[entry["packed_name"]] = candidate_packed.contiguous()
        if entry["replace_scale"]:
            if tuple(candidate_scale.shape) != tuple(
                tensors[entry["target_scale_name"]].shape
            ):
                raise ValueError(f"shape changed for {entry['target_scale_name']}")
            tensors[entry["target_scale_name"]] = candidate_scale.contiguous()
        row = {
            **entry,
            "target_shard": target_shard,
            "shape": list(reference.shape),
            "groups": reference.numel() // 16,
            "values": reference.numel(),
            "global_divisor": float(global_divisor),
            "elapsed_seconds": time.time() - matrix_started,
            "search": search,
        }
        rows.append(row)
        report_handle.write(json.dumps(row, sort_keys=True) + "\n")
        report_handle.flush()
        print(
            json.dumps(
                {
                    "target_shard": target_shard,
                    "matrix": position,
                    "matrices": len(entries),
                    "layer": entry["layer"],
                    "expert": entry["expert"],
                    "projection": entry["projection"],
                    "elapsed_seconds": row["elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del source_weight, source_scale, reference, global_divisor

    save_file(tensors, temporary, metadata=metadata)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    os.replace(temporary, destination)
    os.chmod(destination, 0o644)
    return {
        "matrices": len(entries),
        "elapsed_seconds": time.time() - shard_started,
        "bytes": destination.stat().st_size,
        "summary": summarize_search(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--target-shards",
        type=parse_csv,
        help="Comma-separated Red Hat shard basenames; default is all.",
    )
    parser.add_argument("--scale-radius-below", type=int, default=16)
    parser.add_argument("--scale-radius-above", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--matrix-limit-per-shard",
        type=int,
        help="Developer smoke-test limit; omit for a complete checkpoint build.",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    source_index = load_index(args.source_root)
    target_index = load_index(args.target_root)
    plan = packed_plan(source_index, target_index)
    if args.target_shards:
        unknown = set(args.target_shards) - set(plan)
        if unknown:
            raise SystemExit(f"unknown target shards: {sorted(unknown)}")
        plan = {name: plan[name] for name in args.target_shards}
    if args.matrix_limit_per_shard is not None:
        if args.matrix_limit_per_shard < 1:
            parser.error("--matrix-limit-per-shard must be positive")
        plan = {
            name: entries[: args.matrix_limit_per_shard]
            for name, entries in plan.items()
        }
    summary = plan_summary(plan)
    if args.plan_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.output_root is None:
        parser.error("--output-root is required unless --plan-only is used")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "nvfp4-build-state.json"
    report_path = output_root / "nvfp4-build-tensors.jsonl"
    state = load_state(state_path)
    state["source_root"] = str(args.source_root)
    state["target_root"] = str(args.target_root)
    state["scale_radius_below"] = args.scale_radius_below
    state["scale_radius_above"] = args.scale_radius_above
    state["row_chunk"] = args.row_chunk
    state["plan"] = summary
    atomic_json(state_path, state)

    device = torch.device(args.device)
    with report_path.open("a") as report_handle:
        for target_shard, entries in plan.items():
            if target_shard in state["completed_shards"]:
                print(json.dumps({"target_shard": target_shard, "status": "skipped"}))
                continue
            missing_source = sorted(
                {
                    row[key]
                    for row in entries
                    for key in ("source_weight_shard", "source_scale_shard")
                    if not (args.source_root / row[key]).is_file()
                },
                key=shard_number,
            )
            if missing_source:
                raise FileNotFoundError(
                    "missing official source shards: " + ",".join(missing_source)
                )
            result = build_shard(
                source_root=args.source_root,
                source_index=source_index,
                target_root=args.target_root,
                target_index=target_index,
                output_root=output_root,
                target_shard=target_shard,
                entries=entries,
                device=device,
                scale_radius_below=args.scale_radius_below,
                scale_radius_above=args.scale_radius_above,
                row_chunk=args.row_chunk,
                report_handle=report_handle,
            )
            state["completed_shards"][target_shard] = result
            state["updated_at"] = time.time()
            atomic_json(state_path, state)
            print(json.dumps({"target_shard": target_shard, "status": "complete", **result}))


if __name__ == "__main__":
    main()
