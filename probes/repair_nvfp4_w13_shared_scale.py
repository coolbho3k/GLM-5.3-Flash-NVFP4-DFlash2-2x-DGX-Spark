#!/usr/bin/env python3
"""Restore Marlin-compatible shared W1/W3 scales from a known-good checkpoint.

The compressed-tensors fused-MoE loader concatenates routed-expert gate_proj
and up_proj weights into W13, but the Marlin contract carries one FP32 global
divisor for the fused pair.  A checkpoint that independently optimizes those
two divisors is therefore not represented faithfully at runtime.

This repair deliberately does no new quantization.  It copies gate_proj and
up_proj weight_packed, weight_scale, and weight_global_scale tensors from the
pre-global-divisor checkpoint.  All other tensors, including optimized
down_proj tensors, remain byte-for-byte numerically unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


W13_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj)\."
    r"(?P<kind>weight_packed|weight_scale|weight_global_scale)$"
)


def load_index(root: Path) -> dict[str, str]:
    return json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]


def parse_csv(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one shard")
    return result


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def target_names_by_shard(index: dict[str, str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for name, shard in index.items():
        if W13_PATTERN.match(name):
            result[shard].append(name)
    return {
        shard: sorted(names)
        for shard, names in sorted(result.items())
    }


def validate_indexes(
    current_index: dict[str, str],
    reference_index: dict[str, str],
    output_index: dict[str, str],
) -> None:
    if output_index != current_index:
        raise ValueError("output tensor index differs from current checkpoint")
    missing = sorted(
        name
        for name in current_index
        if W13_PATTERN.match(name) and name not in reference_index
    )
    if missing:
        raise KeyError(f"reference checkpoint is missing W1/W3 tensors: {missing[:8]}")


def replace_shard(
    *,
    current_root: Path,
    reference_root: Path,
    output_root: Path,
    current_index: dict[str, str],
    reference_index: dict[str, str],
    shard: str,
    names: list[str],
) -> dict[str, Any]:
    started = time.time()
    current_path = current_root / shard
    output_path = output_root / shard
    temporary = output_path.with_name(f".{output_path.name}.w13-repair.tmp")

    with safe_open(current_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}

    reference_groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        reference_groups[reference_index[name]].append(name)

    changed = 0
    for reference_shard, reference_names in sorted(reference_groups.items()):
        with safe_open(
            reference_root / reference_shard, framework="pt", device="cpu"
        ) as reference:
            for name in reference_names:
                replacement = reference.get_tensor(name).clone()
                existing = tensors[name]
                if replacement.shape != existing.shape or replacement.dtype != existing.dtype:
                    raise ValueError(f"tensor contract differs for {name}")
                changed += int(not torch.equal(replacement, existing))
                tensors[name] = replacement

    save_file(tensors, temporary, metadata=metadata)
    os.replace(temporary, output_path)
    os.chmod(output_path, 0o644)
    return {
        "tensors_restored": len(names),
        "tensors_changed": changed,
        "bytes": output_path.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }


def validate_restored_content(
    output_root: Path,
    reference_root: Path,
    output_index: dict[str, str],
    reference_index: dict[str, str],
    plan: dict[str, list[str]],
) -> int:
    checked = 0
    for output_shard, names in plan.items():
        reference_groups: dict[str, list[str]] = defaultdict(list)
        for name in names:
            reference_groups[reference_index[name]].append(name)
        with safe_open(
            output_root / output_shard, framework="pt", device="cpu"
        ) as output:
            for reference_shard, reference_names in reference_groups.items():
                with safe_open(
                    reference_root / reference_shard,
                    framework="pt",
                    device="cpu",
                ) as reference:
                    for name in reference_names:
                        if not torch.equal(output.get_tensor(name), reference.get_tensor(name)):
                            raise AssertionError(f"restored tensor differs from reference: {name}")
                        checked += 1
    return checked


def validate_shared_global_scales(
    root: Path, index: dict[str, str]
) -> dict[str, int]:
    values: dict[tuple[int, int, str], torch.Tensor] = {}
    names_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in index.items():
        match = W13_PATTERN.match(name)
        if match and match.group("kind") == "weight_global_scale":
            names_by_shard[shard].append(name)
    for shard, names in names_by_shard.items():
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for name in names:
                match = W13_PATTERN.match(name)
                assert match is not None
                key = (
                    int(match.group("layer")),
                    int(match.group("expert")),
                    match.group("projection"),
                )
                values[key] = handle.get_tensor(name).clone()

    pairs = 0
    mismatches: list[tuple[int, int]] = []
    layer_experts = sorted({(layer, expert) for layer, expert, _ in values})
    for layer, expert in layer_experts:
        gate = values.get((layer, expert, "gate_proj"))
        up = values.get((layer, expert, "up_proj"))
        if gate is None or up is None:
            raise AssertionError(f"incomplete W1/W3 pair at layer={layer} expert={expert}")
        pairs += 1
        if not torch.equal(gate, up):
            mismatches.append((layer, expert))
    if mismatches:
        raise AssertionError(
            f"{len(mismatches)} W1/W3 global-scale pairs still differ; "
            f"first={mismatches[:8]}"
        )
    return {"pairs_checked": pairs, "mismatches": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-shards", type=parse_csv)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    roots = {path.resolve() for path in (args.current_root, args.reference_root, args.output_root)}
    if len(roots) != 3:
        parser.error("current, reference, and output roots must be distinct")

    current_index = load_index(args.current_root)
    reference_index = load_index(args.reference_root)
    output_index = load_index(args.output_root)
    validate_indexes(current_index, reference_index, output_index)
    full_plan = target_names_by_shard(current_index)
    plan = full_plan
    if args.target_shards:
        unknown = set(args.target_shards) - set(full_plan)
        if unknown:
            parser.error(f"unknown target shards: {sorted(unknown)}")
        plan = {shard: full_plan[shard] for shard in args.target_shards}

    state_path = args.output_root / "w13-shared-scale-repair-state.json"
    state = {
        "schema": 1,
        "algorithm": "restore-pre-global-divisor-w13",
        "current_root": str(args.current_root),
        "reference_root": str(args.reference_root),
        "completed_shards": {},
    }
    if state_path.is_file():
        saved = json.loads(state_path.read_text())
        if saved.get("schema") != 1 or saved.get("algorithm") != state["algorithm"]:
            raise ValueError(f"incompatible repair state: {state_path}")
        state = saved

    if not args.verify_only:
        for shard, names in plan.items():
            if shard in state["completed_shards"]:
                print(json.dumps({"shard": shard, "status": "skipped"}), flush=True)
                continue
            result = replace_shard(
                current_root=args.current_root,
                reference_root=args.reference_root,
                output_root=args.output_root,
                current_index=current_index,
                reference_index=reference_index,
                shard=shard,
                names=names,
            )
            state["completed_shards"][shard] = result
            state["updated_at"] = time.time()
            atomic_json(state_path, state)
            print(json.dumps({"shard": shard, "status": "complete", **result}), flush=True)

    checked = validate_restored_content(
        args.output_root, args.reference_root, output_index, reference_index, plan
    )
    shared = validate_shared_global_scales(args.output_root, output_index)
    result = {
        "status": "verified",
        "restored_tensors_checked": checked,
        **shared,
    }
    atomic_json(args.output_root / "w13-shared-scale-repair-verification.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
