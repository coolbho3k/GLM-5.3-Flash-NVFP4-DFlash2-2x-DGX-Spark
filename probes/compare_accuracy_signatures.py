#!/usr/bin/env python3
"""Compare paired student/teacher logprob and DFlash signature artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


EPS = 1e-12


def distribution(row: dict[str, Any]) -> dict[str, float]:
    probs = {
        token: math.exp(float(logprob))
        for token, logprob in row["top_logprobs"].items()
    }
    total = sum(probs.values())
    probs["__TOPK_TAIL__"] = max(1.0 - total, EPS)
    normalizer = sum(probs.values())
    return {key: value / normalizer for key, value in probs.items()}


def js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    l = {key: left.get(key, EPS) for key in keys}
    r = {key: right.get(key, EPS) for key in keys}
    lnorm = sum(l.values())
    rnorm = sum(r.values())
    l = {key: value / lnorm for key, value in l.items()}
    r = {key: value / rnorm for key, value in r.items()}
    middle = {key: 0.5 * (l[key] + r[key]) for key in keys}
    kl_l = sum(l[key] * math.log(l[key] / middle[key]) for key in keys)
    kl_r = sum(r[key] * math.log(r[key] / middle[key]) for key in keys)
    return 0.5 * (kl_l + kl_r)


def reported_top1(row: dict[str, Any]) -> str:
    top = row["top_logprobs"]
    if not top:
        return row["token_key"]
    return max(top, key=top.get)


def paired_positions(
    student: list[dict[str, Any]], teacher: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for index, (srow, trow) in enumerate(zip(student, teacher)):
        sdist = distribution(srow)
        tdist = distribution(trow)
        rows.append(
            {
                "position": index,
                "same_sampled_token": srow["token_key"] == trow["token_key"],
                "same_top1": reported_top1(srow) == reported_top1(trow),
                "js_nats_topk": js_divergence(sdist, tdist),
                "teacher_token_student_logprob": srow["top_logprobs"].get(
                    trow["token_key"]
                ),
            }
        )
        if srow["token_key"] != trow["token_key"]:
            break
    return rows


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    student = json.loads(args.student.read_text())
    teacher = json.loads(args.teacher.read_text())
    if student["corpus_sha256"] != teacher["corpus_sha256"]:
        raise SystemExit("student and teacher corpus hashes differ")
    skey = {(row["id"], row["repeat"]): row for row in student["cases"]}
    tkey = {(row["id"], row["repeat"]): row for row in teacher["cases"]}
    if skey.keys() != tkey.keys():
        raise SystemExit("student and teacher case sets differ")

    cases = []
    all_positions = []
    for key in sorted(skey):
        srow, trow = skey[key], tkey[key]
        positions = paired_positions(srow["logprobs"], trow["logprobs"])
        all_positions.extend(positions)
        cases.append(
            {
                "id": key[0],
                "repeat": key[1],
                "category": srow["category"],
                "student_quality_pass": srow["quality_pass"],
                "teacher_quality_pass": trow["quality_pass"],
                "exact_output_match": srow["content"] == trow["content"],
                "common_prefix_positions": sum(
                    1 for row in positions if row["same_sampled_token"]
                ),
                "compared_positions": len(positions),
                "mean_js_nats_topk": mean(
                    [row["js_nats_topk"] for row in positions]
                ),
                "student_acceptance_ratio": srow["spec_decode"][
                    "acceptance_ratio"
                ],
                "teacher_acceptance_ratio": trow["spec_decode"][
                    "acceptance_ratio"
                ],
                "positions": positions,
            }
        )
    report = {
        "schema": 1,
        "student": student["label"],
        "teacher": teacher["label"],
        "corpus_sha256": student["corpus_sha256"],
        "summary": {
            "pairs": len(cases),
            "student_quality_pass_rate": mean(
                [float(row["student_quality_pass"]) for row in cases]
            ),
            "teacher_quality_pass_rate": mean(
                [float(row["teacher_quality_pass"]) for row in cases]
            ),
            "exact_output_match_rate": mean(
                [float(row["exact_output_match"]) for row in cases]
            ),
            "top1_agreement_rate_common_prefix": mean(
                [float(row["same_top1"]) for row in all_positions]
            ),
            "mean_js_nats_topk_common_prefix": mean(
                [row["js_nats_topk"] for row in all_positions]
            ),
            "student_acceptance_ratio": mean(
                [
                    row["student_acceptance_ratio"]
                    for row in cases
                    if row["student_acceptance_ratio"] is not None
                ]
            ),
            "teacher_acceptance_ratio": mean(
                [
                    row["teacher_acceptance_ratio"]
                    for row in cases
                    if row["teacher_acceptance_ratio"] is not None
                ]
            ),
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
