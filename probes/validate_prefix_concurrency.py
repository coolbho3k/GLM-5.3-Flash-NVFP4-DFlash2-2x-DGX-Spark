#!/usr/bin/env python3
"""Validate concurrent reuse of a shared, multi-page cached prefix."""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from validate_kv_candidate import chat, content, metric


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=6000)
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    filler = "amber forest silver mountain " * args.repetitions
    midpoint = len(filler) // 2
    context = (
        filler[:midpoint]
        + " The unique shared verification code is MAGNOLIA. "
        + filler[midpoint:]
    )
    system = {
        "role": "system",
        "content": "Shared reference text follows. Preserve it for lookup.\n" + context,
    }

    warm_response, warm_seconds = chat(
        base_url,
        [
            system,
            {
                "role": "user",
                "content": "Return the unique shared verification code only.",
            },
        ],
        max_tokens=16,
    )
    warm_answer = content(warm_response).strip()
    if warm_answer != "MAGNOLIA":
        raise AssertionError(f"warm request returned {warm_answer!r}")
    prompt_tokens = int((warm_response.get("usage") or {}).get("prompt_tokens") or 0)
    if prompt_tokens < 8192:
        raise AssertionError(f"shared prefix is only {prompt_tokens} tokens")

    barrier = threading.Barrier(args.concurrency)

    def request(index: int) -> dict[str, Any]:
        barrier.wait(timeout=30)
        response, elapsed = chat(
            base_url,
            [
                system,
                {
                    "role": "user",
                    "content": (
                        f"Concurrent request {index}: return the unique shared "
                        "verification code only."
                    ),
                },
            ],
            max_tokens=16,
        )
        return {
            "index": index,
            "answer": content(response).strip(),
            "seconds": round(elapsed, 3),
        }

    metric_name = "vllm:prefix_cache_hits_total"
    before = metric(base_url, metric_name)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        requests = list(executor.map(request, range(args.concurrency)))
    after = metric(base_url, metric_name)

    failures = [item for item in requests if item["answer"] != "MAGNOLIA"]
    hit_tokens = int(after - before)
    report = {
        "ok": not failures and hit_tokens > 0,
        "concurrency": args.concurrency,
        "prompt_tokens": prompt_tokens,
        "prefix_hit_tokens": hit_tokens,
        "warm_seconds": round(warm_seconds, 3),
        "requests": requests,
        "failures": failures,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
