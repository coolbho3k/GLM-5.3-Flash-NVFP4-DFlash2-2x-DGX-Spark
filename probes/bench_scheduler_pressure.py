#!/usr/bin/env python3
"""Measure decode responsiveness while cold prefills enter at C2-C6."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_prefill_mixed import (
    decode_payload,
    health,
    prefill_once,
    stream_chat,
)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_decoders(results: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    rates = [float(result["decode_tokens_per_second"]) for result in results]
    gaps = [
        float(result["max_stream_gap_seconds"])
        for result in results
    ]
    p95_gaps = [
        float(result["p95_stream_gap_seconds"])
        for result in results
    ]
    total_tokens = sum(int(result["completion_tokens"]) for result in results)
    return {
        "count": len(results),
        "wall_seconds": round(wall_seconds, 6),
        "aggregate_tokens_per_second": round(total_tokens / wall_seconds, 6),
        "mean_per_stream_tokens_per_second": round(statistics.mean(rates), 6),
        "minimum_per_stream_tokens_per_second": round(min(rates), 6),
        "p50_ttft_seconds": round(
            statistics.median(float(result["ttft_seconds"]) for result in results),
            6,
        ),
        "maximum_ttft_seconds": round(
            max(float(result["ttft_seconds"]) for result in results), 6
        ),
        "p95_of_stream_p95_gap_seconds": round(percentile(p95_gaps, 0.95), 6),
        "maximum_stream_gap_seconds": round(max(gaps), 6),
        "streams": results,
    }


def decode_wave(
    base_url: str,
    *,
    decoder_count: int,
    decode_tokens: int,
    case_prefix: str,
    corpus_seed: str,
    ready_events: list[threading.Event] | None = None,
) -> tuple[list[threading.Thread], list[dict[str, Any]], float]:
    boxes: list[dict[str, Any]] = [{} for _ in range(decoder_count)]
    events = ready_events or [threading.Event() for _ in range(decoder_count)]
    started = time.perf_counter()

    def worker(index: int) -> None:
        try:
            boxes[index]["result"] = stream_chat(
                base_url,
                decode_payload(
                    decode_tokens,
                    case_id=f"{case_prefix}-decoder-{index}",
                    corpus_seed=corpus_seed,
                ),
                on_first_token=events[index].set,
            )
        except BaseException as exc:
            boxes[index]["error"] = exc
            events[index].set()

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(decoder_count)
    ]
    for thread in threads:
        thread.start()
    return threads, boxes, started


def finish_decode_wave(
    threads: list[threading.Thread],
    boxes: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    for thread in threads:
        thread.join(timeout=1800)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("decoder wave did not finish")
    for box in boxes:
        if "error" in box:
            raise box["error"]
    results = [box["result"] for box in boxes]
    # The caller may wait for prefill before joining finished decoders. Count
    # only the decoder wave's lifetime, not that extra wait.
    return summarize_decoders(
        results, max(result["finished_monotonic"] for result in results) - started
    )


def prefill_wave(
    base_url: str,
    *,
    prefill_count: int,
    prefill_tokens: int,
    case_prefix: str,
    corpus_seed: str,
) -> dict[str, Any]:
    boxes: list[dict[str, Any]] = [{} for _ in range(prefill_count)]
    started = time.perf_counter()

    def worker(index: int) -> None:
        try:
            boxes[index]["result"] = prefill_once(
                base_url,
                prefill_tokens,
                case_id=f"{case_prefix}-prefill-{index}",
                corpus_seed=corpus_seed,
                phrase_index=index + 1,
            )
        except BaseException as exc:
            boxes[index]["error"] = exc

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(prefill_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1800)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("prefill wave did not finish")
    for box in boxes:
        if "error" in box:
            raise box["error"]
    results = [box["result"] for box in boxes]
    wall_seconds = time.perf_counter() - started
    return {
        "count": prefill_count,
        "wall_seconds": round(wall_seconds, 6),
        "aggregate_input_tokens_per_second": round(
            sum(int(result["prompt_tokens"]) for result in results) / wall_seconds,
            6,
        ),
        "maximum_ttft_seconds": round(
            max(float(result["ttft_seconds"]) for result in results), 6
        ),
        "streams": results,
    }


def run_case(
    base_url: str,
    *,
    decoder_count: int,
    prefill_count: int,
    decode_tokens: int,
    prefill_tokens: int,
    corpus_seed: str,
    profile_mixed: bool = False,
) -> dict[str, Any]:
    # Keep decoder prompts identical between control/mixed waves and across
    # scheduler profiles. DFlash acceptance is content-sensitive, while the
    # short decoder prefix is too small for caching to affect the comparison.
    decoder_seed = "glm53-scheduler-pressure-decode-v1"
    decoder_case = f"c{decoder_count}-decoder"
    from bench_serving_quick import metrics, profile_session
    before_control = metrics(base_url)
    control_threads, control_boxes, control_started = decode_wave(
        base_url,
        decoder_count=decoder_count,
        decode_tokens=decode_tokens,
        case_prefix=decoder_case,
        corpus_seed=decoder_seed,
    )
    control = finish_decode_wave(control_threads, control_boxes, control_started)
    after_control = metrics(base_url)
    control["metrics_delta"] = {
        key: after_control[key] - before_control[key] for key in before_control
    }

    ready = [threading.Event() for _ in range(decoder_count)]
    mixed_threads, mixed_boxes, mixed_started = decode_wave(
        base_url,
        decoder_count=decoder_count,
        decode_tokens=decode_tokens,
        case_prefix=decoder_case,
        corpus_seed=decoder_seed,
        ready_events=ready,
    )
    for event in ready:
        if not event.wait(timeout=180):
            raise RuntimeError("not all mixed decoders emitted a first token")
    for box in mixed_boxes:
        if "error" in box:
            raise box["error"]

    with profile_session(base_url, profile_mixed):
        prefills = prefill_wave(
            base_url,
            prefill_count=prefill_count,
            prefill_tokens=prefill_tokens,
            case_prefix=f"c{decoder_count}-mixed",
            corpus_seed=corpus_seed,
        )
    mixed = finish_decode_wave(mixed_threads, mixed_boxes, mixed_started)
    after_mixed = metrics(base_url)
    mixed["metrics_delta"] = {
        key: after_mixed[key] - after_control[key] for key in after_control
    }
    prefill_start = min(s["started_monotonic"] for s in prefills["streams"])
    prefill_end = max(s["first_token_monotonic"] for s in prefills["streams"])
    first_decoder_end = min(s["finished_monotonic"] for s in mixed["streams"])
    full_concurrency_overlap = max(
        0.0, min(prefill_end, first_decoder_end) - prefill_start
    )
    return {
        "decoder_count": decoder_count,
        "prefill_count": prefill_count,
        "total_sessions": decoder_count + prefill_count,
        "control_decode": control,
        "mixed_decode": mixed,
        "concurrent_prefill": prefills,
        "prefill_fraction_at_full_decode_concurrency": round(
            full_concurrency_overlap / max(prefill_end - prefill_start, 1e-9), 6
        ),
        "mixed_over_control_mean_per_stream_tps": round(
            mixed["mean_per_stream_tokens_per_second"]
            / control["mean_per_stream_tokens_per_second"],
            6,
        ),
        "mixed_over_control_aggregate_tps": round(
            mixed["aggregate_tokens_per_second"]
            / control["aggregate_tokens_per_second"],
            6,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus-seed", required=True)
    parser.add_argument("--decoder-counts", default="1,3,5")
    parser.add_argument("--prefills-per-wave", type=int, default=1)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--prefill-tokens", type=int, default=16384)
    parser.add_argument("--profile-mixed", action="store_true",
                        help="Capture bounded traces once mixed decoding is active")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    health(base_url)
    decoder_counts = [
        int(value) for value in args.decoder_counts.split(",") if value.strip()
    ]
    if any(count <= 0 for count in decoder_counts):
        raise ValueError("decoder counts must be positive")
    if any(count + args.prefills_per_wave > 6 for count in decoder_counts):
        raise ValueError("each wave must fit the six-session server limit")

    # Warm the streaming path without sharing prefixes with measured cases.
    stream_chat(
        base_url,
        decode_payload(8, case_id="warmup", corpus_seed=args.corpus_seed),
    )

    report: dict[str, Any] = {
        "schema": 1,
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "url": base_url,
        "decode_tokens": args.decode_tokens,
        "prefill_tokens": args.prefill_tokens,
        "prefills_per_wave": args.prefills_per_wave,
        "corpus_seed": args.corpus_seed,
        "profile_mixed": args.profile_mixed,
        "cases": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for decoder_count in decoder_counts:
        case = run_case(
            base_url,
            decoder_count=decoder_count,
            prefill_count=args.prefills_per_wave,
            decode_tokens=args.decode_tokens,
            prefill_tokens=args.prefill_tokens,
            corpus_seed=args.corpus_seed,
            profile_mixed=args.profile_mixed,
        )
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
