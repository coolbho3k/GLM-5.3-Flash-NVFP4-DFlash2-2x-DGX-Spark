#!/usr/bin/env python3
"""Capture comparable output logits and DFlash acceptance from a vLLM server.

The artifact is intentionally self-contained: it records the corpus hash,
request payloads, top-logprob distributions, exact output, latency, quality
checks, and before/after speculative-decoding counters. Run the same corpus
against the RedHat target and the EXL3 teacher, then compare with
``compare_accuracy_signatures.py``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


def http_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def http_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode()


def metric_value(text: str, name: str) -> float:
    values = []
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            values.append(float(line.rsplit(" ", 1)[1]))
    return sum(values)


def metrics(url: str) -> dict[str, Any]:
    text = http_text(url.rstrip("/") + "/metrics")
    result: dict[str, Any] = {
        key: metric_value(text, name) for key, name in COUNTERS.items()
    }
    per_position: dict[str, float] = {}
    prefix = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
    for line in text.splitlines():
        if not line.startswith(prefix + "{"):
            continue
        match = re.search(r'position="([0-9]+)"', line)
        if match:
            per_position[match.group(1)] = float(line.rsplit(" ", 1)[1])
    result["accepted_per_position"] = per_position
    return result


def subtract(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    result = {key: after[key] - before[key] for key in COUNTERS}
    positions = set(after["accepted_per_position"]) | set(
        before["accepted_per_position"]
    )
    result["accepted_per_position"] = {
        key: after["accepted_per_position"].get(key, 0.0)
        - before["accepted_per_position"].get(key, 0.0)
        for key in sorted(positions, key=int)
    }
    drafted = result["draft_tokens"]
    result["acceptance_ratio"] = (
        result["accepted_tokens"] / drafted if drafted > 0 else None
    )
    drafts = result["drafts"]
    result["mean_acceptance_length"] = (
        1.0 + result["accepted_tokens"] / drafts if drafts > 0 else None
    )
    return result


def token_key(item: dict[str, Any]) -> str:
    raw = item.get("bytes")
    if raw is not None:
        return "b64:" + base64.b64encode(bytes(raw)).decode()
    return "str:" + item["token"]


def normalize_logprobs(choice: dict[str, Any]) -> list[dict[str, Any]]:
    content = (choice.get("logprobs") or {}).get("content") or []
    rows = []
    for item in content:
        top = {
            token_key(candidate): float(candidate["logprob"])
            for candidate in item.get("top_logprobs", [])
        }
        rows.append(
            {
                "token": item.get("token"),
                "token_key": token_key(item),
                "logprob": float(item["logprob"]),
                "top_logprobs": top,
            }
        )
    return rows


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not case.get("id") or not case.get("messages"):
            raise ValueError(f"{path}:{number}: missing id/messages")
        filler = case.pop("filler", None)
        if filler:
            replacement = filler["text"] * int(filler["repetitions"])
            for message in case["messages"]:
                message["content"] = message["content"].replace(
                    "{FILLER}", replacement
                )
        cases.append(case)
    return cases


def capture_case(
    base_url: str,
    model: str,
    case: dict[str, Any],
    repeat: int,
    top_logprobs: int,
) -> dict[str, Any]:
    before = metrics(base_url)
    payload = {
        "model": model,
        "messages": case["messages"],
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "max_tokens": int(case.get("max_tokens", 256)),
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    started = time.perf_counter()
    response = http_json(base_url.rstrip("/") + "/v1/chat/completions", payload)
    elapsed = time.perf_counter() - started
    after = metrics(base_url)
    choice = response["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    pattern = case.get("expected_regex")
    quality_pass = bool(re.search(pattern, content.strip())) if pattern else None
    usage = response.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "id": case["id"],
        "category": case.get("category", "unspecified"),
        "repeat": repeat,
        "request": payload,
        "content": content,
        "reasoning": message.get("reasoning"),
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": completion_tokens / elapsed if elapsed else math.inf,
        "quality_regex": pattern,
        "quality_pass": quality_pass,
        "logprobs": normalize_logprobs(choice),
        "spec_decode": subtract(after, before),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).parents[1] / "accuracy-campaign/corpus.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    raw_corpus = args.corpus.read_bytes()
    cases = load_cases(args.corpus)
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in cases}
        if missing:
            raise SystemExit(f"unknown cases: {sorted(missing)}")
    model_info = http_json(args.url.rstrip("/") + "/v1/models")
    report: dict[str, Any] = {
        "schema": 1,
        "label": args.label,
        "url": args.url,
        "model": args.model,
        "model_info": model_info,
        "corpus_sha256": hashlib.sha256(raw_corpus).hexdigest(),
        "top_logprobs": args.top_logprobs,
        "repeats": args.repeats,
        "created_unix": time.time(),
        "cases": [],
    }
    for repeat in range(args.repeats):
        for case in cases:
            row = capture_case(
                args.url, args.model, case, repeat, args.top_logprobs
            )
            report["cases"].append(row)
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "repeat": repeat,
                        "quality_pass": row["quality_pass"],
                        "tok_s": round(row["output_tokens_per_second"], 3),
                        "acceptance": row["spec_decode"]["acceptance_ratio"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
