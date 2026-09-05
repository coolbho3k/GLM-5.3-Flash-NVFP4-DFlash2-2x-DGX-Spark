#!/usr/bin/env python3
"""Summarize the matched off/m64/off/m64 serving screen; fail on incomplete data."""
import argparse
import json
from pathlib import Path
import statistics


def analyze(root, prefix="fat-native", control_mode="off", candidate_mode="m64", single_workload_pass=False):
    assert control_mode != candidate_mode
    labels = ("A", "B", "A2", "B2")
    workload_labels = ("A", "B") if single_workload_pass else labels
    data = {
        label: {kind: json.loads((root / f"{prefix}-{label}-{kind}.json").read_text())
                for kind in (("prefill", "decode", "mixed") if label in workload_labels else ("prefill",))}
        for label in labels
    }
    prefills = [data[label]["prefill"]["cases"][0] for label in labels]
    assert all(len(data[label]["prefill"]["cases"]) == 1 for label in labels)
    assert len({p["prompt_sha256"] for p in prefills}) == 1
    assert len({p["prompt_tokens"] for p in prefills}) == 1
    assert len({p["cache_salt"] for p in prefills}) == 4
    assert all(p["cache_salt"] and p["global_prefix_hit_tokens_delta"] == 0 for p in prefills)
    assert all(p["content_preview"].strip() == "OK" for p in prefills)
    result = {"order": list(labels), "modes": {control_mode: ["A", "A2"], candidate_mode: ["B", "B2"]},
              "prefill": {}, "decode": [], "mixed": {}}
    if single_workload_pass:
        result["decode_and_mixed_labels"] = list(workload_labels)
    reference_config = data["A"]["decode"]["config"]
    for label in workload_labels:
        report = data[label]["decode"]
        for field in ("mode", "levels", "prompts", "max_tokens", "rounds", "profile"):
            assert report["config"].get(field) == reference_config.get(field), (label, field)
        assert len(report["cases"]) == 6
        assert {(c["concurrency"], c["kind"]) for c in report["cases"]} == {
            (c, k) for c in (1, 3, 6) for k in ("prose", "code")}
        for case in report["cases"]:
            metrics = case["metrics_delta"]
            assert metrics["request_success_total"] == case["concurrency"]
            assert metrics["num_preemptions_total"] == 0
            assert all(s["completion_tokens"] == reference_config["max_tokens"]
                       for s in case["streams"])
    for mode, members in result["modes"].items():
        samples = [data[label]["prefill"]["cases"][0]["input_tokens_per_second"] for label in members]
        result["prefill"][mode] = {"input_tps": samples, "median_input_tps": statistics.median(samples)}
    result["prefill"]["gain_percent"] = 100 * (
        result["prefill"][candidate_mode]["median_input_tps"] / result["prefill"][control_mode]["median_input_tps"] - 1)
    for concurrency in (1, 3, 6):
        for kind in ("prose", "code"):
            row = {"concurrency": concurrency, "kind": kind}
            for mode, members in result["modes"].items():
                cases = [c for label in members if label in workload_labels for c in data[label]["decode"]["cases"]
                         if c["concurrency"] == concurrency and c["kind"] == kind]
                metrics = [c["metrics_delta"] for c in cases]
                row[mode] = {
                    "aggregate_tps": [c["aggregate_tokens_per_second"] for c in cases],
                    "median_aggregate_tps": statistics.median(c["aggregate_tokens_per_second"] for c in cases),
                    "accepted_fraction": sum(m["spec_decode_num_accepted_tokens_total"] for m in metrics)
                        / sum(m["spec_decode_num_draft_tokens_total"] for m in metrics),
                }
                if concurrency == 1:
                    row[mode]["wall_ms_per_draft"] = 1000 * sum(c["wall_seconds"] for c in cases) / sum(
                        m["spec_decode_num_drafts_total"] for m in metrics)
            result["decode"].append(row)
    mixed_prompts, mixed_salts = set(), set()
    for label in workload_labels:
        report = data[label]["mixed"]
        assert report["cold_prefill"] and not report["profile_mixed"]
        assert len(report["cases"]) == 1
        case = report["cases"][0]
        assert case["decoder_count"] == 3 and case["prefill_count"] == 1
        metrics = case["mixed_decode"]["metrics_delta"]
        assert metrics["request_success_total"] == 4 and metrics["num_preemptions_total"] == 0
        stream = case["concurrent_prefill"]["streams"][0]
        mixed_prompts.add(stream["prompt_sha256"])
        mixed_salts.add(stream["cache_salt"])
        result["mixed"][label] = {
            "decode_per_stream_tps": case["mixed_decode"]["mean_per_stream_tokens_per_second"],
            "prefill_input_tps": case["concurrent_prefill"]["aggregate_input_tokens_per_second"],
            "full_concurrency_overlap": case["prefill_fraction_at_full_decode_concurrency"],
            "maximum_stream_gap_seconds": case["mixed_decode"]["maximum_stream_gap_seconds"],
            "accepted_fraction": metrics["spec_decode_num_accepted_tokens_total"] / metrics["spec_decode_num_draft_tokens_total"],
        }
    assert len(mixed_prompts) == 1 and len(mixed_salts) == len(workload_labels) and None not in mixed_salts
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prefix", default="fat-native")
    parser.add_argument("--control-mode", default="off")
    parser.add_argument("--candidate-mode", default="m64")
    parser.add_argument("--single-workload-pass", action="store_true",
                        help="A2/B2 repeat prefill only; do not imply repeated decode/mixed measurements")
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory, args.prefix, args.control_mode, args.candidate_mode,
                             args.single_workload_pass), indent=2))
