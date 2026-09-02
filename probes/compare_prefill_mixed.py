#!/usr/bin/env python3
"""Compare matched Marlin and FlashInfer B12X serving benchmark reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def geometric_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute geometric mean of an empty list")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def mean_metric(rows: list[dict[str, Any]], section: str, metric: str) -> float:
    return statistics.mean(float(row[section][metric]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    if baseline["corpus_seed"] != candidate["corpus_seed"]:
        raise ValueError("reports use different deterministic corpora")

    baseline_pure = {
        int(row["target_prompt_tokens"]): row for row in baseline["pure_prefill"]
    }
    candidate_pure = {
        int(row["target_prompt_tokens"]): row for row in candidate["pure_prefill"]
    }
    if baseline_pure.keys() != candidate_pure.keys():
        raise ValueError("reports have different pure-prefill target sets")

    pure: list[dict[str, Any]] = []
    pure_ratios: list[float] = []
    for target in sorted(baseline_pure):
        old = baseline_pure[target]
        new = candidate_pure[target]
        if int(old["prompt_tokens"]) != int(new["prompt_tokens"]):
            raise ValueError(
                f"target {target} used different tokenized prompts: "
                f"{old['prompt_tokens']} vs {new['prompt_tokens']}"
            )
        ratio = (
            float(new["input_tokens_per_second"])
            / float(old["input_tokens_per_second"])
        )
        pure_ratios.append(ratio)
        pure.append(
            {
                "target_prompt_tokens": target,
                "baseline_prompt_tokens": old["prompt_tokens"],
                "candidate_prompt_tokens": new["prompt_tokens"],
                "baseline_input_tokens_per_second": old[
                    "input_tokens_per_second"
                ],
                "candidate_input_tokens_per_second": new[
                    "input_tokens_per_second"
                ],
                "candidate_over_baseline": round(ratio, 6),
            }
        )

    baseline_mixed = baseline["mixed"]
    candidate_mixed = candidate["mixed"]
    if len(baseline_mixed) != len(candidate_mixed) or not baseline_mixed:
        raise ValueError("reports have incompatible mixed-workload rounds")

    baseline_control_tps = mean_metric(
        baseline_mixed, "control_decode", "decode_tokens_per_second"
    )
    candidate_control_tps = mean_metric(
        candidate_mixed, "control_decode", "decode_tokens_per_second"
    )
    baseline_mixed_tps = mean_metric(
        baseline_mixed, "mixed_decode", "decode_tokens_per_second"
    )
    candidate_mixed_tps = mean_metric(
        candidate_mixed, "mixed_decode", "decode_tokens_per_second"
    )
    baseline_prefill_tps = mean_metric(
        baseline_mixed, "concurrent_prefill", "input_tokens_per_second"
    )
    candidate_prefill_tps = mean_metric(
        candidate_mixed, "concurrent_prefill", "input_tokens_per_second"
    )
    baseline_max_gap = mean_metric(
        baseline_mixed, "mixed_decode", "max_stream_gap_seconds"
    )
    candidate_max_gap = mean_metric(
        candidate_mixed, "mixed_decode", "max_stream_gap_seconds"
    )

    summary = {
        "schema": 1,
        "baseline_label": baseline["label"],
        "candidate_label": candidate["label"],
        "pure_prefill": pure,
        "pure_prefill_geomean_candidate_over_baseline": round(
            geometric_mean(pure_ratios), 6
        ),
        "decode_only_candidate_over_baseline": round(
            candidate_control_tps / baseline_control_tps, 6
        ),
        "mixed_decode_candidate_over_baseline": round(
            candidate_mixed_tps / baseline_mixed_tps, 6
        ),
        "mixed_prefill_candidate_over_baseline": round(
            candidate_prefill_tps / baseline_prefill_tps, 6
        ),
        "mixed_decode_interference": {
            "baseline_mixed_over_control": round(
                baseline_mixed_tps / baseline_control_tps, 6
            ),
            "candidate_mixed_over_control": round(
                candidate_mixed_tps / candidate_control_tps, 6
            ),
        },
        "mean_mixed_max_stream_gap_seconds": {
            "baseline": round(baseline_max_gap, 6),
            "candidate": round(candidate_max_gap, 6),
        },
    }
    summary["recommend_candidate"] = bool(
        summary["pure_prefill_geomean_candidate_over_baseline"] >= 1.03
        and summary["decode_only_candidate_over_baseline"] >= 0.97
        and summary["mixed_decode_candidate_over_baseline"] >= 0.97
    )

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
