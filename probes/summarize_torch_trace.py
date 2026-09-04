#!/usr/bin/env python3
"""Summarize PyTorch Chrome traces without loading torch or using the GPU."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def union_duration(intervals):
    total = 0.0
    end = float("-inf")
    for start, finish in sorted(intervals):
        total += max(0.0, finish - max(start, end))
        end = max(end, finish)
    return total


def summarize(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    groups = defaultdict(lambda: [0, 0.0])
    intervals = []
    steps = []
    for event in trace["traceEvents"]:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        name = event.get("name", "")
        duration = float(event.get("dur", 0))
        start = float(event.get("ts", 0))
        if category in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime"):
            groups[(category, name)][0] += 1
            groups[(category, name)][1] += duration
        if category in ("kernel", "gpu_memcpy", "gpu_memset"):
            intervals.append((start, start + duration))
        if name.startswith("ProfilerStep"):
            steps.append({"name": name, "milliseconds": duration / 1000})
    rows = [{"category": cat, "name": name, "calls": count,
             "total_ms": duration / 1000,
             "mean_us": duration / count}
            for (cat, name), (count, duration) in groups.items()]
    span = max((end for _, end in intervals), default=0) - min(
        (start for start, _ in intervals), default=0)
    busy = union_duration(intervals)
    return {"path": str(path), "gpu_span_ms": span / 1000,
            "gpu_active_union_ms": busy / 1000,
            "gpu_gap_fraction": 1 - busy / span if span else None,
            "steps": steps,
            "gpu": sorted((r for r in rows if r["category"] != "cuda_runtime"),
                          key=lambda r: -r["total_ms"]),
            "cuda_runtime": sorted((r for r in rows if r["category"] == "cuda_runtime"),
                                   key=lambda r: -r["total_ms"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summaries = [summarize(path) for path in args.traces]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summaries, indent=2) + "\n")
    for summary in summaries:
        print(summary["path"])
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ("gpu", "cuda_runtime", "path")}))
        for category in ("gpu", "cuda_runtime"):
            print(category)
            for row in summary[category][:20]:
                print(f'{row["total_ms"]:10.3f} ms {row["calls"]:6d} calls '
                      f'{row["mean_us"]:9.2f} us {row["name"][:180]}')


if __name__ == "__main__":
    main()
