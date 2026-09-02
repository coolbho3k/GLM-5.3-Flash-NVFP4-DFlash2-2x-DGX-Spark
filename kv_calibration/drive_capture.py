#!/usr/bin/env python3
"""Drive an explicitly-enabled calibration server with a JSONL prompt corpus.

This program is intentionally not part of server startup. Run it only during
an approved capture window after starting the tooling image with capture
environment variables and a shared control/output directory.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
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


def _remote_json(destination: str, value: dict) -> None:
    """Atomically update HOST:/absolute/path without exposing shell input."""
    if ":" not in destination:
        raise ValueError("--remote-control must be HOST:/absolute/path")
    host, path_text = destination.split(":", 1)
    if not host or not path_text.startswith("/"):
        raise ValueError("--remote-control must be HOST:/absolute/path")
    path = Path(path_text)
    encoded = base64.b64encode((json.dumps(value) + "\n").encode()).decode()
    temporary = path.with_name(f".{path.name}.tmp")
    command = (
        f"mkdir -p {shlex.quote(str(path.parent))} && "
        f"printf %s {shlex.quote(encoded)} | base64 -d > "
        f"{shlex.quote(str(temporary))} && "
        f"mv {shlex.quote(str(temporary))} {shlex.quote(str(path))}"
    )
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        check=True,
    )


def _set_control(path: Path, remote: str | None, value: dict) -> None:
    _atomic_json(path, value)
    if remote:
        _remote_json(remote, value)


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
    parser.add_argument(
        "--remote-control",
        help="Optional HOST:/absolute/path control file for the second rank",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-final-flush", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    rows = rows[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]
    successes = 0
    active_bucket: str | None = None
    try:
        for index, row in enumerate(rows, start=args.start):
            bucket = str(row.get("bucket", "unlabeled"))
            if bucket != active_bucket:
                _set_control(
                    args.control,
                    args.remote_control,
                    {"bucket": bucket, "phase": "auto", "enabled": True},
                )
                active_bucket = bucket
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
        flush_epoch = 0
        try:
            if successes and not args.no_final_flush:
                flush_epoch = time.time_ns()
                _set_control(
                    args.control,
                    args.remote_control,
                    {
                        "bucket": "capture-flush",
                        "phase": "auto",
                        "enabled": True,
                        "flush_epoch": flush_epoch,
                    },
                )
                time.sleep(0.35)
                try:
                    _request(
                        args.url,
                        args.api_key,
                        {
                            "model": args.model,
                            "messages": [
                                {"role": "user", "content": "Reply with OK."}
                            ],
                            "max_tokens": 1,
                            "temperature": 0,
                            "stream": False,
                        },
                        args.timeout,
                    )
                    print("final capture flush completed")
                except (urllib.error.URLError, TimeoutError) as error:
                    print(f"final capture flush request failed: {error}")
        finally:
            _set_control(
                args.control,
                args.remote_control,
                {
                    "bucket": "disabled",
                    "phase": "auto",
                    "enabled": False,
                    "flush_epoch": flush_epoch,
                },
            )
    print(f"completed {successes}/{len(rows)} requests")


if __name__ == "__main__":
    main()
