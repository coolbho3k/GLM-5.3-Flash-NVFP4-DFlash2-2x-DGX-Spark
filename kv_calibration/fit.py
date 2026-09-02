#!/usr/bin/env python3
"""Fit per-layer outer scales from bounded NVFP4 latent capture shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .numerics import evaluate_groups


SCHEMA = "glm-nvfp4-mla-gscale-v1"
ALGORITHM = "nvfp4-e2m1-e4m3-four-over-six"
RECORD_LAYOUT = "glm-zero-rope-288b"


def _read_shard(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as shard:
        metadata = json.loads(str(shard["metadata"].item()))
        values = np.asarray(shard["values"], dtype=np.float32)
        priorities = np.asarray(shard["priorities"], dtype=np.float64)
    if metadata.get("schema") != "glm-nvfp4-mla-capture-v1":
        raise ValueError(f"unsupported capture schema in {path}")
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError(f"invalid values shape in {path}: {values.shape}")
    if priorities.shape != (values.shape[0],):
        raise ValueError(f"invalid priorities shape in {path}: {priorities.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite captured values in {path}")
    return metadata, values, priorities


def load_capture(
    roots: list[Path], *, per_stratum_limit: int
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    sources: list[dict[str, Any]] = []
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(sorted(root.rglob("*.npz")))
        else:
            raise FileNotFoundError(root)
    if not paths:
        raise ValueError("no .npz capture shards found")

    for path in sorted(set(paths)):
        metadata, values, priorities = _read_shard(path)
        layer = str(metadata["layer"])
        stratum = str(metadata["stratum"])
        grouped.setdefault((layer, stratum), []).append((values, priorities))
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "layer": layer,
                "stratum": stratum,
                "rows": int(values.shape[0]),
                "seen": int(metadata.get("seen", values.shape[0])),
            }
        )

    result: dict[str, dict[str, np.ndarray]] = {}
    for (layer, stratum), parts in grouped.items():
        values = np.concatenate([part[0] for part in parts], axis=0)
        priorities = np.concatenate([part[1] for part in parts], axis=0)
        if len(values) > per_stratum_limit:
            keep = np.argpartition(priorities, -per_stratum_limit)[-per_stratum_limit:]
            values = values[keep]
        result.setdefault(layer, {})[stratum] = values
    return result, sources


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return seed ^ int.from_bytes(digest[:8], "little")


def split_layer(
    strata: dict[str, np.ndarray], *, holdout_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    train: list[np.ndarray] = []
    holdout: list[np.ndarray] = []
    coverage: dict[str, dict[str, int]] = {}
    for name, values in sorted(strata.items()):
        rng = np.random.default_rng(_stable_seed(seed, name))
        order = rng.permutation(len(values))
        holdout_count = max(1, int(round(len(values) * holdout_fraction)))
        if len(values) > 1:
            holdout_count = min(holdout_count, len(values) - 1)
        else:
            holdout_count = 0
        holdout_values = values[order[:holdout_count]]
        train_values = values[order[holdout_count:]]
        train.append(train_values)
        if holdout_count:
            holdout.append(holdout_values)
        coverage[name] = {
            "train_groups": int(len(train_values)),
            "heldout_groups": int(len(holdout_values)),
        }
    if not train or sum(len(x) for x in train) == 0:
        raise ValueError("capture has no training groups")
    return (
        np.concatenate(train),
        np.concatenate(holdout) if holdout else np.empty((0, 16), np.float32),
        coverage,
    )


def _objective(
    metrics: dict[str, float | int],
    *,
    kind: str,
    baseline: dict[str, float | int],
    blend_relative_weight: float,
) -> float:
    if kind == "mse":
        return float(metrics["mse"])
    if kind == "group-relative":
        return float(metrics["mean_group_relative_mse"])
    if kind == "blend":
        mse = float(metrics["mse"]) / max(float(baseline["mse"]), 1e-300)
        relative = float(metrics["mean_group_relative_mse"]) / max(
            float(baseline["mean_group_relative_mse"]), 1e-300
        )
        return mse + blend_relative_weight * relative
    raise ValueError(kind)


def fit_layer(
    train: np.ndarray,
    heldout: np.ndarray,
    *,
    global_scales: np.ndarray,
    objective: str,
    blend_relative_weight: float,
    chunk_rows: int,
    heldout_gate: bool,
) -> dict[str, Any]:
    baseline_train = evaluate_groups(train, latent_scale=1.0, chunk_rows=chunk_rows)
    curve: list[dict[str, Any]] = []
    best: tuple[float, float, dict[str, float | int]] | None = None
    for global_scale in global_scales:
        latent_scale = 1.0 / float(global_scale)
        metrics = evaluate_groups(
            train, latent_scale=latent_scale, chunk_rows=chunk_rows
        )
        score = _objective(
            metrics,
            kind=objective,
            baseline=baseline_train,
            blend_relative_weight=blend_relative_weight,
        )
        curve.append(
            {
                "global_scale": float(global_scale),
                "latent_scale": latent_scale,
                "objective": score,
                "mse": float(metrics["mse"]),
                "nmse": float(metrics["nmse"]),
                "mean_group_relative_mse": float(
                    metrics["mean_group_relative_mse"]
                ),
                "zero_scale_fraction": float(metrics["zero_scale_fraction"]),
                "saturated_scale_fraction": float(
                    metrics["saturated_scale_fraction"]
                ),
            }
        )
        candidate = (score, abs(np.log2(global_scale)), metrics)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
            selected_global_scale = float(global_scale)

    assert best is not None
    selected_latent_scale = 1.0 / selected_global_scale
    baseline_heldout = (
        evaluate_groups(heldout, latent_scale=1.0, chunk_rows=chunk_rows)
        if len(heldout)
        else None
    )
    selected_heldout = (
        evaluate_groups(
            heldout, latent_scale=selected_latent_scale, chunk_rows=chunk_rows
        )
        if len(heldout)
        else None
    )
    gated = False
    if heldout_gate and baseline_heldout is not None and selected_heldout is not None:
        selected_score = _objective(
            selected_heldout,
            kind=objective,
            baseline=baseline_heldout,
            blend_relative_weight=blend_relative_weight,
        )
        baseline_score = _objective(
            baseline_heldout,
            kind=objective,
            baseline=baseline_heldout,
            blend_relative_weight=blend_relative_weight,
        )
        if selected_score > baseline_score:
            selected_global_scale = 1.0
            selected_latent_scale = 1.0
            selected_heldout = baseline_heldout
            gated = True

    selected_train = evaluate_groups(
        train, latent_scale=selected_latent_scale, chunk_rows=chunk_rows
    )
    return {
        "global_scale": selected_global_scale,
        "latent_scale": selected_latent_scale,
        "heldout_gate_fell_back_to_one": gated,
        "train": {"baseline": baseline_train, "selected": selected_train},
        "heldout": {
            "baseline": baseline_heldout,
            "selected": selected_heldout,
        },
        "curve": curve,
    }


def apply_cross_validation_gate(
    result: dict[str, Any],
    validation_holdouts: list[tuple[str, np.ndarray]],
    *,
    objective: str,
    blend_relative_weight: float,
    chunk_rows: int,
    minimum_improvement_percent: float,
) -> bool:
    """Fall back to G=1 unless every independent holdout clears the gate."""
    candidate_latent_scale = float(result["latent_scale"])
    validations: list[dict[str, Any]] = []
    passed = True
    for label, heldout in validation_holdouts:
        baseline = evaluate_groups(
            heldout, latent_scale=1.0, chunk_rows=chunk_rows
        )
        selected = evaluate_groups(
            heldout,
            latent_scale=candidate_latent_scale,
            chunk_rows=chunk_rows,
        )
        baseline_objective = _objective(
            baseline,
            kind=objective,
            baseline=baseline,
            blend_relative_weight=blend_relative_weight,
        )
        selected_objective = _objective(
            selected,
            kind=objective,
            baseline=baseline,
            blend_relative_weight=blend_relative_weight,
        )
        improvement = 100.0 * (
            baseline_objective - selected_objective
        ) / max(abs(baseline_objective), 1e-300)
        source_passed = (
            candidate_latent_scale == 1.0
            or improvement >= minimum_improvement_percent
        )
        passed = passed and source_passed
        validations.append(
            {
                "label": label,
                "heldout_groups": int(len(heldout)),
                "baseline": baseline,
                "candidate": selected,
                "objective_improvement_percent": improvement,
                "passed": source_passed,
            }
        )

    result["cross_validation"] = validations
    result["cross_validation_gate_fell_back_to_one"] = (
        bool(validation_holdouts)
        and candidate_latent_scale != 1.0
        and not passed
    )
    if result["cross_validation_gate_fell_back_to_one"]:
        result["pre_cross_validation_global_scale"] = result["global_scale"]
        result["pre_cross_validation_latent_scale"] = result["latent_scale"]
        result["global_scale"] = 1.0
        result["latent_scale"] = 1.0
        result["train"]["selected"] = result["train"]["baseline"]
        result["heldout"]["selected"] = result["heldout"]["baseline"]
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=("mse", "group-relative", "blend"), default="mse")
    parser.add_argument("--blend-relative-weight", type=float, default=0.10)
    parser.add_argument("--log2-global-min", type=float, default=-4.0)
    parser.add_argument("--log2-global-max", type=float, default=16.0)
    parser.add_argument("--steps-per-octave", type=int, default=8)
    parser.add_argument("--per-stratum-limit", type=int, default=65536)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--chunk-rows", type=int, default=32768)
    parser.add_argument("--allow-heldout-regression", action="store_true")
    parser.add_argument(
        "--cross-validation-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "independently gate the merged candidate on this capture root; "
            "repeat for each node or capture population"
        ),
    )
    parser.add_argument(
        "--minimum-cross-validation-improvement-percent",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    if args.steps_per_octave <= 0:
        parser.error("--steps-per-octave must be positive")
    if not 0.0 < args.holdout_fraction < 1.0:
        parser.error("--holdout-fraction must be between zero and one")
    if args.per_stratum_limit <= 1:
        parser.error("--per-stratum-limit must exceed one")
    if args.minimum_cross_validation_improvement_percent < 0.0:
        parser.error("--minimum-cross-validation-improvement-percent must be nonnegative")
    exponent_count = int(
        round((args.log2_global_max - args.log2_global_min) * args.steps_per_octave)
    )
    exponents = args.log2_global_min + np.arange(exponent_count + 1) / args.steps_per_octave
    global_scales = np.unique(np.append(np.exp2(exponents), 1.0))

    capture, sources = load_capture(
        args.captures, per_stratum_limit=args.per_stratum_limit
    )
    cross_captures: list[tuple[str, dict[str, dict[str, np.ndarray]], int]] = []
    for root in args.cross_validation_root:
        cross_capture, cross_sources = load_capture(
            [root], per_stratum_limit=args.per_stratum_limit
        )
        cross_captures.append(
            (str(root), cross_capture, len(cross_sources))
        )

    layers: dict[str, Any] = {}
    for layer, strata in sorted(capture.items()):
        train, heldout, coverage = split_layer(
            strata,
            holdout_fraction=args.holdout_fraction,
            seed=_stable_seed(args.seed, layer),
        )
        result = fit_layer(
            train,
            heldout,
            global_scales=global_scales,
            objective=args.objective,
            blend_relative_weight=args.blend_relative_weight,
            chunk_rows=args.chunk_rows,
            heldout_gate=not args.allow_heldout_regression,
        )
        result["coverage"] = coverage
        validation_holdouts: list[tuple[str, np.ndarray]] = []
        for label, cross_capture, _ in cross_captures:
            if layer not in cross_capture:
                raise ValueError(
                    f"cross-validation root {label!r} has no data for {layer}"
                )
            _, cross_heldout, _ = split_layer(
                cross_capture[layer],
                holdout_fraction=args.holdout_fraction,
                seed=_stable_seed(args.seed, layer),
            )
            validation_holdouts.append((label, cross_heldout))
        apply_cross_validation_gate(
            result,
            validation_holdouts,
            objective=args.objective,
            blend_relative_weight=args.blend_relative_weight,
            chunk_rows=args.chunk_rows,
            minimum_improvement_percent=(
                args.minimum_cross_validation_improvement_percent
            ),
        )
        layers[layer] = result
        print(
            f"{layer}: G={result['global_scale']:.8g} "
            f"L={result['latent_scale']:.8g} "
            f"train MSE {result['train']['baseline']['mse']:.8g} -> "
            f"{result['train']['selected']['mse']:.8g}"
        )

    artifact = {
        "schema": SCHEMA,
        "algorithm": ALGORITHM,
        "record_layout": RECORD_LAYOUT,
        "scale_convention": {
            "global_scale": "G multiplies the local scale before E4M3 encoding",
            "latent_scale": "L=1/G multiplies decoded E4M3 scales in the reader",
        },
        "fit": {
            "objective": args.objective,
            "blend_relative_weight": args.blend_relative_weight,
            "log2_global_min": args.log2_global_min,
            "log2_global_max": args.log2_global_max,
            "steps_per_octave": args.steps_per_octave,
            "holdout_fraction": args.holdout_fraction,
            "seed": args.seed,
            "heldout_gate": not args.allow_heldout_regression,
        },
        "cross_validation_gate": {
            "roots": [
                {"path": label, "source_shards": source_count}
                for label, _, source_count in cross_captures
            ],
            "minimum_improvement_percent": (
                args.minimum_cross_validation_improvement_percent
            ),
        },
        "sources": sources,
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
