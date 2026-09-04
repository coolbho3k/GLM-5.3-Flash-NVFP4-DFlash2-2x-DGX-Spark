#!/usr/bin/env python3
"""Optimize NVFP4 group-16 scales/rounding against an FP8 or BF16 source.

The Red Hat checkpoint uses memoryless min/max: each 16-value group scale is
derived from amax/6 and rounded to FP8 E4M3. This tool keeps the checkpoint's
global divisor and exact compressed-tensors layout, but searches nearby
representable FP8 group scales and chooses the E2M1 reconstruction with minimum
weighted squared error. It emits only replacement weight_packed/weight_scale
tensors plus a self-contained numerical report.

Run inside the serving image, which already contains torch, safetensors, and
the exact FP8/FP4 dtype implementations used by vLLM.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


E2M1 = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
E2M1_THRESHOLDS = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0))
SOURCE_SUFFIXES = {
    "down_proj": ("down_proj.weight", "down_proj.weight_scale_inv"),
    "gate_proj": ("gate_proj.weight", "gate_proj.weight_scale_inv"),
    "up_proj": ("up_proj.weight", "up_proj.weight_scale_inv"),
}
SOURCE_FORMATS = ("block-fp8", "bf16")


def parse_csv_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def load_index(root: Path) -> dict[str, str]:
    path = root / "model.safetensors.index.json"
    return json.loads(path.read_text())["weight_map"]


def load_tensor(
    root: Path,
    index: dict[str, str],
    name: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    shard = root / index[name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
    return tensor.to(device)


def source_names(
    layer: int,
    expert: int,
    projection: str,
    source_format: str = "block-fp8",
) -> tuple[str, str | None]:
    if source_format not in SOURCE_FORMATS:
        raise ValueError(f"unsupported source format: {source_format}")
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
    weight, scale = SOURCE_SUFFIXES[projection]
    return prefix + weight, (prefix + scale if source_format == "block-fp8" else None)


def target_names(layer: int, expert: int, projection: str) -> tuple[str, str, str]:
    prefix = (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}."
    )
    return (
        prefix + "weight_packed",
        prefix + "weight_scale",
        prefix + "weight_global_scale",
    )


def dequant_source(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantize official 128x128 block-FP8 expert weights."""
    rows, cols = weight.shape
    expected = (math.ceil(rows / 128), math.ceil(cols / 128))
    if tuple(scale_inv.shape) != expected:
        raise ValueError(
            f"source scale shape {tuple(scale_inv.shape)} != expected {expected}"
        )
    scale = scale_inv.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return weight.float() * scale[:rows, :cols].float()


def source_reference(
    weight: torch.Tensor,
    scale_inv: torch.Tensor | None,
    *,
    source_format: str,
) -> torch.Tensor:
    """Return the FP32 reference matrix for a supported source checkpoint.

    BF16 is deliberately strict. This prevents an FP8 checkpoint from being
    interpreted as unscaled values when the caller accidentally selects the
    wrong source format.
    """
    if source_format == "bf16":
        if scale_inv is not None:
            raise ValueError("BF16 source must not provide weight_scale_inv")
        if weight.dtype != torch.bfloat16:
            raise ValueError(
                f"BF16 source expected torch.bfloat16 expert weight, got {weight.dtype}"
            )
        return weight.float()
    if source_format == "block-fp8":
        if scale_inv is None:
            raise ValueError("block-FP8 source requires weight_scale_inv")
        return dequant_source(weight, scale_inv)
    raise ValueError(f"unsupported source format: {source_format}")


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    lookup = E2M1.to(packed.device)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    magnitude = lookup[(nibbles & 0x07).long()]
    return torch.where((nibbles & 0x08).bool(), -magnitude, magnitude)


def dequant_target(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_divisor: torch.Tensor,
) -> torch.Tensor:
    values = unpack_fp4(packed)
    effective = scale.float() / global_divisor.float()
    return values * effective.repeat_interleave(16, dim=1)


def nearest_e2m1_magnitude(scaled_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    thresholds = E2M1_THRESHOLDS.to(scaled_abs.device)
    indices = torch.zeros_like(scaled_abs, dtype=torch.uint8)
    for threshold in thresholds:
        indices += (scaled_abs > threshold).to(torch.uint8)
    values = E2M1.to(scaled_abs.device)[indices.long()]
    return indices, values


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[1] % 2:
        raise ValueError("FP4 packing requires an even K dimension")
    paired = codes.reshape(codes.shape[0], -1, 2)
    return (paired[..., 0] | (paired[..., 1] << 4)).to(torch.uint8)


@torch.no_grad()
def optimize_groups(
    weight: torch.Tensor,
    global_divisor: torch.Tensor,
    *,
    scale_radius_below: int,
    scale_radius_above: int,
    row_chunk: int,
    input_second_moment: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Search nearby representable E4M3 scale codes independently per group."""
    rows, cols = weight.shape
    if cols % 16:
        raise ValueError(f"K={cols} is not divisible by group size 16")
    groups = cols // 16
    output_packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device="cpu")
    output_scale = torch.empty(
        (rows, groups), dtype=torch.float8_e4m3fn, device="cpu"
    )
    offsets = torch.arange(
        -scale_radius_below,
        scale_radius_above + 1,
        device=weight.device,
        dtype=torch.int16,
    )
    delta_codes: list[torch.Tensor] = []
    clipped_values = 0
    total_values = rows * cols

    moment = None
    if input_second_moment is not None:
        if tuple(input_second_moment.shape) != (cols,):
            raise ValueError(
                f"input second moment shape {tuple(input_second_moment.shape)} "
                f"does not match K={cols}"
            )
        moment = input_second_moment.float().reshape(1, groups, 16, 1)

    for row_start in range(0, rows, row_chunk):
        row_end = min(row_start + row_chunk, rows)
        chunk = weight[row_start:row_end].float().reshape(-1, groups, 16)
        amax = chunk.abs().amax(dim=2)
        ideal_stored_scale = (amax / 6.0) * global_divisor.float()
        baseline_scale = ideal_stored_scale.to(torch.float8_e4m3fn)
        baseline_codes = baseline_scale.view(torch.uint8).to(torch.int16)

        candidate_codes = baseline_codes.unsqueeze(-1) + offsets
        candidate_codes.clamp_(1, 126)
        candidate_scales = (
            candidate_codes.to(torch.uint8)
            .view(torch.float8_e4m3fn)
            .float()
            / global_divisor.float()
        )
        scaled_abs = chunk.abs().unsqueeze(-1) / candidate_scales.unsqueeze(2)
        _, quantized_magnitude = nearest_e2m1_magnitude(scaled_abs)
        reconstructed_abs = quantized_magnitude * candidate_scales.unsqueeze(2)
        squared_error = (reconstructed_abs - chunk.abs().unsqueeze(-1)).square()
        if moment is not None:
            squared_error *= moment
        candidate_error = squared_error.sum(dim=2)
        best_index = candidate_error.argmin(dim=-1)
        best_codes = candidate_codes.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
        best_scale = best_codes.to(torch.uint8).view(torch.float8_e4m3fn)
        effective_scale = best_scale.float() / global_divisor.float()

        scaled = chunk.abs() / effective_scale.unsqueeze(-1)
        magnitude_codes, _ = nearest_e2m1_magnitude(scaled)
        clipped_values += int((scaled > 5.0).sum().item())
        sign_codes = torch.signbit(chunk).to(torch.uint8) << 3
        fp4_codes = (magnitude_codes | sign_codes).reshape(row_end - row_start, cols)

        output_packed[row_start:row_end].copy_(pack_codes(fp4_codes).cpu())
        output_scale[row_start:row_end].copy_(best_scale.cpu())
        delta_codes.append((best_codes - baseline_codes).cpu())

    deltas = torch.cat([item.flatten() for item in delta_codes])
    unique_deltas, delta_counts = torch.unique(deltas, return_counts=True)
    delta_histogram = {
        str(int(delta)): int(count)
        for delta, count in zip(unique_deltas, delta_counts, strict=True)
    }
    stats = {
        "scale_code_delta_min": int(deltas.min()),
        "scale_code_delta_max": int(deltas.max()),
        "scale_code_delta_mean": float(deltas.float().mean()),
        "scale_code_changed_fraction": float((deltas != 0).float().mean()),
        "lower_boundary_fraction": float(
            (deltas == -scale_radius_below).float().mean()
        ),
        "upper_boundary_fraction": float(
            (deltas == scale_radius_above).float().mean()
        ),
        "scale_code_delta_histogram": delta_histogram,
        "clipped_value_fraction": clipped_values / total_values,
        "candidate_count": int(offsets.numel()),
    }
    return output_packed, output_scale, stats


@torch.no_grad()
def select_global_divisor(
    weight: torch.Tensor,
    global_divisor: torch.Tensor,
    *,
    steps_per_octave: int,
    search_rows: int,
    heldout_tolerance: float,
    scale_radius_below: int,
    scale_radius_above: int,
    row_chunk: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select the E4M3-grid phase of a matrix's FP32 global divisor.

    Multiplying the divisor by powers of two is nearly redundant because E4M3
    repeats by octave. Searching one octave therefore finds the useful phase
    without changing the tensor layout. Alternating, evenly spread rows form
    deterministic train/held-out sets; a held-out regression falls back to the
    checkpoint divisor.
    """
    original = global_divisor.detach().clone()
    if steps_per_octave <= 0:
        return original, {
            "enabled": False,
            "original_divisor": float(original),
            "selected_divisor": float(original),
            "selected_multiplier": 1.0,
            "fell_back": False,
        }
    if search_rows < 2:
        raise ValueError("global-divisor search needs at least two rows")
    if heldout_tolerance < 0:
        raise ValueError("held-out tolerance must be non-negative")

    count = min(weight.shape[0], search_rows)
    indices = torch.linspace(
        0, weight.shape[0] - 1, count, device=weight.device
    ).round().long().unique()
    train = weight.index_select(0, indices[::2])
    heldout_indices = indices[1::2]
    heldout = (
        weight.index_select(0, heldout_indices)
        if heldout_indices.numel()
        else train
    )
    phase_start = -(steps_per_octave // 2)
    exponents = range(phase_start, phase_start + steps_per_octave)
    multipliers = sorted(
        {1.0, *(2.0 ** (offset / steps_per_octave) for offset in exponents)}
    )

    def score(rows: torch.Tensor, divisor: torch.Tensor) -> dict[str, float]:
        packed, scale, _ = optimize_groups(
            rows,
            divisor,
            scale_radius_below=scale_radius_below,
            scale_radius_above=scale_radius_above,
            row_chunk=min(row_chunk, rows.shape[0]),
            input_second_moment=None,
        )
        reconstructed = dequant_target(
            packed.to(weight.device), scale.to(weight.device), divisor
        )
        delta_power = (reconstructed - rows.float()).square().mean()
        reference_power = rows.float().square().mean()
        result = {
            "mse": float(delta_power),
            "relative_mse": float(delta_power / reference_power),
        }
        del packed, scale, reconstructed
        return result

    train_curve: list[dict[str, float]] = []
    for multiplier in multipliers:
        divisor = original * multiplier
        metrics = score(train, divisor)
        train_curve.append({"multiplier": multiplier, **metrics})
    selected_train = min(
        train_curve,
        key=lambda row: (row["mse"], abs(math.log2(row["multiplier"]))),
    )
    selected_multiplier = float(selected_train["multiplier"])
    original_heldout = score(heldout, original)
    selected_heldout = score(heldout, original * selected_multiplier)
    fell_back = (
        selected_multiplier != 1.0
        and selected_heldout["mse"]
        > original_heldout["mse"] * (1.0 + heldout_tolerance)
    )
    if fell_back:
        selected_multiplier = 1.0
        selected_heldout = original_heldout
    selected = original * selected_multiplier
    return selected, {
        "enabled": True,
        "steps_per_octave": steps_per_octave,
        "sample_rows": int(indices.numel()),
        "train_rows": int(train.shape[0]),
        "heldout_rows": int(heldout.shape[0]),
        "heldout_tolerance": heldout_tolerance,
        "original_divisor": float(original),
        "selected_divisor": float(selected),
        "selected_multiplier": selected_multiplier,
        "fell_back": fell_back,
        "train_curve": train_curve,
        "original_heldout": original_heldout,
        "selected_heldout": selected_heldout,
    }


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    got = candidate.float()
    delta = got - ref
    ref_power = ref.square().mean()
    return {
        "relative_rmse": float((delta.square().mean() / ref_power).sqrt()),
        "cosine": float(F.cosine_similarity(ref.flatten(), got.flatten(), dim=0)),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
    }


@torch.no_grad()
def output_error(
    reference_weight: torch.Tensor,
    candidate_weight: torch.Tensor,
    *,
    tokens: int,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator(device=reference_weight.device)
    generator.manual_seed(seed)
    inputs = torch.randn(
        (tokens, reference_weight.shape[1]),
        generator=generator,
        device=reference_weight.device,
        dtype=torch.bfloat16,
    )
    reference = F.linear(inputs, reference_weight.to(torch.bfloat16))
    candidate = F.linear(inputs, candidate_weight.to(torch.bfloat16))
    metrics = error_metrics(reference, candidate)
    metrics["tokens"] = tokens
    return metrics


def aggregate(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-format",
        choices=SOURCE_FORMATS,
        default="block-fp8",
        help="Reference checkpoint encoding; block-fp8 preserves the existing workflow.",
    )
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=parse_csv_ints, default=[3, 23, 43])
    parser.add_argument("--experts", type=parse_csv_ints, default=[0])
    parser.add_argument(
        "--projections",
        default="gate_proj,up_proj,down_proj",
        help="Comma-separated projection names.",
    )
    parser.add_argument("--scale-radius-below", type=int, default=16)
    parser.add_argument("--scale-radius-above", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=64)
    parser.add_argument(
        "--global-divisor-steps-per-octave",
        type=int,
        default=0,
        help="Search this many divisor phases in one octave; 0 disables it.",
    )
    parser.add_argument("--global-divisor-search-rows", type=int, default=32)
    parser.add_argument(
        "--global-divisor-heldout-tolerance", type=float, default=0.0
    )
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    projections = [item for item in args.projections.split(",") if item]
    unknown = set(projections) - set(SOURCE_SUFFIXES)
    if unknown:
        raise SystemExit(f"unknown projections: {sorted(unknown)}")
    source_index = load_index(args.source_root)
    target_index = load_index(args.target_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    replacements: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    started = time.time()
    for layer in args.layers:
        for expert in args.experts:
            for projection in projections:
                source_weight_name, source_scale_name = source_names(
                    layer, expert, projection, args.source_format
                )
                packed_name, target_scale_name, global_name = target_names(
                    layer, expert, projection
                )
                print(
                    json.dumps(
                        {
                            "layer": layer,
                            "expert": expert,
                            "projection": projection,
                            "status": "loading",
                        }
                    ),
                    flush=True,
                )
                source_weight = load_tensor(
                    args.source_root, source_index, source_weight_name, device=device
                )
                source_scale = (
                    load_tensor(
                        args.source_root,
                        source_index,
                        source_scale_name,
                        device=device,
                    )
                    if source_scale_name is not None
                    else None
                )
                reference = source_reference(
                    source_weight,
                    source_scale,
                    source_format=args.source_format,
                )
                del source_weight, source_scale

                target_packed = load_tensor(
                    args.target_root, target_index, packed_name, device=device
                )
                target_scale = load_tensor(
                    args.target_root, target_index, target_scale_name, device=device
                )
                global_divisor = load_tensor(
                    args.target_root, target_index, global_name, device=device
                )
                baseline = dequant_target(
                    target_packed, target_scale, global_divisor
                )
                selected_divisor, divisor_search = select_global_divisor(
                    reference,
                    global_divisor,
                    steps_per_octave=args.global_divisor_steps_per_octave,
                    search_rows=args.global_divisor_search_rows,
                    heldout_tolerance=args.global_divisor_heldout_tolerance,
                    scale_radius_below=args.scale_radius_below,
                    scale_radius_above=args.scale_radius_above,
                    row_chunk=args.row_chunk,
                )
                candidate_packed, candidate_scale, search_stats = optimize_groups(
                    reference,
                    selected_divisor,
                    scale_radius_below=args.scale_radius_below,
                    scale_radius_above=args.scale_radius_above,
                    row_chunk=args.row_chunk,
                    input_second_moment=None,
                )
                candidate = dequant_target(
                    candidate_packed.to(device),
                    candidate_scale.to(device),
                    selected_divisor,
                )
                baseline_weight_error = error_metrics(reference, baseline)
                candidate_weight_error = error_metrics(reference, candidate)
                baseline_output_error = output_error(
                    reference,
                    baseline,
                    tokens=args.output_tokens,
                    seed=args.seed + layer * 1000 + expert * 10,
                )
                candidate_output_error = output_error(
                    reference,
                    candidate,
                    tokens=args.output_tokens,
                    seed=args.seed + layer * 1000 + expert * 10,
                )
                row = {
                    "layer": layer,
                    "expert": expert,
                    "projection": projection,
                    "shape": list(reference.shape),
                    "source_shard": source_index[source_weight_name],
                    "source_scale_shard": (
                        source_index[source_scale_name]
                        if source_scale_name is not None
                        else None
                    ),
                    "target_shard": target_index[packed_name],
                    "original_global_divisor": float(global_divisor),
                    "selected_global_divisor": float(selected_divisor),
                    "global_divisor_multiplier": (
                        float(selected_divisor) / float(global_divisor)
                    ),
                    "global_divisor_search": divisor_search,
                    "baseline_weight_error": baseline_weight_error,
                    "candidate_weight_error": candidate_weight_error,
                    "baseline_output_error": baseline_output_error,
                    "candidate_output_error": candidate_output_error,
                    "relative_rmse_improvement": (
                        1.0
                        - candidate_weight_error["relative_rmse"]
                        / baseline_weight_error["relative_rmse"]
                    ),
                    "output_rmse_improvement": (
                        1.0
                        - candidate_output_error["relative_rmse"]
                        / baseline_output_error["relative_rmse"]
                    ),
                    "search": search_stats,
                }
                rows.append(row)
                replacements[packed_name] = candidate_packed.contiguous()
                replacements[target_scale_name] = candidate_scale.contiguous()
                if args.global_divisor_steps_per_octave > 0:
                    replacements[global_name] = selected_divisor.cpu().contiguous()
                print(json.dumps(row, sort_keys=True), flush=True)
                del (
                    reference,
                    baseline,
                    candidate,
                    target_packed,
                    target_scale,
                    selected_divisor,
                    global_divisor,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    method = (
        "group16_e2m1_mse_scale_and_global_divisor_search"
        if args.global_divisor_steps_per_octave > 0
        else "group16_e2m1_mse_scale_search"
    )
    tensor_path = args.output_dir / "optimized.safetensors"
    save_file(
        replacements,
        tensor_path,
        metadata={
            "format": "pt",
            "method": method,
            "source": str(args.source_root),
            "source_format": args.source_format,
            "target": str(args.target_root),
        },
    )
    report = {
        "schema": 2,
        "method": method,
        "source_root": str(args.source_root),
        "source_format": args.source_format,
        "target_root": str(args.target_root),
        "layers": args.layers,
        "experts": args.experts,
        "projections": projections,
        "scale_radius_below": args.scale_radius_below,
        "scale_radius_above": args.scale_radius_above,
        "global_divisor_steps_per_octave": args.global_divisor_steps_per_octave,
        "global_divisor_search_rows": args.global_divisor_search_rows,
        "global_divisor_heldout_tolerance": args.global_divisor_heldout_tolerance,
        "output_tokens": args.output_tokens,
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
        "replacement_tensors": sorted(replacements),
        "summary": {
            "tensors": len(rows),
            "global_divisors_changed": sum(
                row["global_divisor_multiplier"] != 1.0 for row in rows
            ),
            "global_divisor_search_fallbacks": sum(
                bool(row["global_divisor_search"]["fell_back"]) for row in rows
            ),
            "baseline_weight_relative_rmse_mean": aggregate(
                [row["baseline_weight_error"] for row in rows], "relative_rmse"
            ),
            "candidate_weight_relative_rmse_mean": aggregate(
                [row["candidate_weight_error"] for row in rows], "relative_rmse"
            ),
            "baseline_output_relative_rmse_mean": aggregate(
                [row["baseline_output_error"] for row in rows], "relative_rmse"
            ),
            "candidate_output_relative_rmse_mean": aggregate(
                [row["candidate_output_error"] for row in rows], "relative_rmse"
            ),
            "relative_rmse_improvement_mean": aggregate(
                rows, "relative_rmse_improvement"
            ),
            "output_rmse_improvement_mean": aggregate(
                rows, "output_rmse_improvement"
            ),
        },
        "tensors": rows,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
