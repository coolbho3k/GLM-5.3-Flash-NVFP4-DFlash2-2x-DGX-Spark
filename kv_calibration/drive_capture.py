#!/usr/bin/env python3
"""Drive an explicitly-enabled calibration server with a JSONL prompt corpus.

This program is intentionally not part of server startup. Run it only during
an approved capture window after starting the tooling image with capture
environment variables and a shared control/output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _request(url: str, api_key: str | None, payload: dict, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    rows = rows[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]
    successes = 0
    try:
        for index, row in enumerate(rows, start=args.start):
            bucket = str(row.get("bucket", "unlabeled"))
            _atomic_json(
                args.control,
                {"bucket": bucket, "phase": "auto", "enabled": True},
            )
            if "messages" in row:
                messages = row["messages"]
            elif "prompt" in row:
                messages = [{"role": "user", "content": row["prompt"]}]
            else:
                raise ValueError(f"corpus row {index} has neither messages nor prompt")
            payload = {
                "model": args.model,
                "messages": messages,
                "max_tokens": int(row.get("max_tokens", args.max_tokens)),
                "temperature": 0,
                "stream": False,
            }
            if "extra_body" in row:
                payload.update(row["extra_body"])
            try:
                _request(args.url, args.api_key, payload, args.timeout)
                successes += 1
                print(f"[{index + 1}/{args.start + len(rows)}] ok bucket={bucket}")
            except (urllib.error.URLError, TimeoutError) as error:
                print(f"[{index + 1}] failed bucket={bucket}: {error}")
            if args.delay:
                time.sleep(args.delay)
    finally:
        _atomic_json(
            args.control,
            {"bucket": "disabled", "phase": "auto", "enabled": False},
        )
    print(f"completed {successes}/{len(rows)} requests")


if __name__ == "__main__":
    main()
