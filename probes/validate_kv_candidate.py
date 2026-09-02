#!/usr/bin/env python3
"""End-to-end acceptance probe for a GLM KV-cache candidate.

The probe deliberately covers paths that a nominal KV slot count does not:
ordinary generation, deterministic arithmetic, tool parsing, image input,
optional video input, multi-page long-context retrieval, and an observable
prefix-cache hit.
It uses only the Python standard library so it can run on either cluster node.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import struct
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 900,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(
        url, data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def chat(
    base_url: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 64,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {
        "model": "glm-5.3-flash",
        "messages": messages,
        "temperature": 0,
        "seed": 123,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    started = time.perf_counter()
    response = request_json(
        "POST", f"{base_url}/v1/chat/completions", payload
    )
    return response, time.perf_counter() - started


def content(response: dict[str, Any]) -> str:
    return str(response["choices"][0]["message"].get("content") or "")


def metric(base_url: str, name: str) -> float:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        text = response.read().decode()
    total = 0.0
    found = False
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            total += float(line.rsplit(maxsplit=1)[-1])
            found = True
    if not found:
        raise RuntimeError(f"metric not found: {name}")
    return total


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    )


def black_png_data_url(width: int = 16, height: int = 16) -> str:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + b"\x00" * (width * 3) for _ in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(scanlines, 9))
        + png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def video_data_url(path: str) -> str:
    video = Path(path).read_bytes()
    if not video:
        raise ValueError(f"empty video file: {path}")
    return "data:video/mp4;base64," + base64.b64encode(video).decode()


def run_case(
    results: dict[str, Any],
    name: str,
    case: Callable[[], dict[str, Any]],
) -> None:
    started = time.perf_counter()
    try:
        detail = case()
        results[name] = {
            "ok": True,
            "seconds": round(time.perf_counter() - started, 3),
            **detail,
        }
    except Exception as exc:
        results[name] = {
            "ok": False,
            "seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps({name: results[name]}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--long-repetitions",
        type=int,
        default=4500,
        help="Four-token filler repetitions; 4500 produces roughly 20K tokens.",
    )
    parser.add_argument(
        "--skip-long", action="store_true", help="Skip long-context/prefix tests."
    )
    parser.add_argument(
        "--min-long-tokens",
        type=int,
        default=8192,
        help="Require at least this many prompt tokens in the retrieval probe.",
    )
    parser.add_argument(
        "--filler-phrase",
        default="granite willow cobalt river ",
        help="Repeated text for the long-context probe; change it for a cold prefix.",
    )
    parser.add_argument(
        "--video-path",
        help="Optional MP4 path; when set, validate the model's video input path.",
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    results: dict[str, Any] = {}

    def health_case() -> dict[str, Any]:
        with urllib.request.urlopen(f"{base_url}/health", timeout=30) as response:
            if response.status != 200:
                raise AssertionError(response.status)
        models = request_json("GET", f"{base_url}/v1/models")
        ids = [entry["id"] for entry in models["data"]]
        if "glm-5.3-flash" not in ids:
            raise AssertionError(ids)
        return {"models": ids}

    run_case(results, "health", health_case)

    def exact_text_case() -> dict[str, Any]:
        response, elapsed = chat(
            base_url,
            [{"role": "user", "content": "Reply with exactly ORCHID and nothing else."}],
            max_tokens=16,
        )
        answer = content(response).strip()
        if answer != "ORCHID":
            raise AssertionError(repr(answer))
        return {"answer": answer, "request_seconds": round(elapsed, 3)}

    run_case(results, "exact_text", exact_text_case)

    def arithmetic_case() -> dict[str, Any]:
        response, elapsed = chat(
            base_url,
            [{"role": "user", "content": "Compute 37 multiplied by 41. Reply with digits only."}],
            max_tokens=16,
        )
        answer = content(response).strip()
        if answer != "1517":
            raise AssertionError(repr(answer))
        return {"answer": answer, "request_seconds": round(elapsed, 3)}

    run_case(results, "arithmetic", arithmetic_case)

    def tool_case() -> dict[str, Any]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Look up current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        response, elapsed = chat(
            base_url,
            [{"role": "user", "content": "Use lookup_weather for Seattle."}],
            max_tokens=96,
            tools=tools,
        )
        message = response["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls or calls[0]["function"]["name"] != "lookup_weather":
            raise AssertionError(message)
        arguments = json.loads(calls[0]["function"]["arguments"])
        if arguments.get("city", "").lower() != "seattle":
            raise AssertionError(arguments)
        return {
            "tool": calls[0]["function"]["name"],
            "arguments": arguments,
            "request_seconds": round(elapsed, 3),
        }

    run_case(results, "tool_call", tool_case)

    def vision_case() -> dict[str, Any]:
        response, elapsed = chat(
            base_url,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": black_png_data_url()},
                        },
                        {
                            "type": "text",
                            "text": "What single color fills this image? Reply with one word.",
                        },
                    ],
                }
            ],
            max_tokens=16,
        )
        answer = content(response).strip()
        if "black" not in answer.lower():
            raise AssertionError(repr(answer))
        return {"answer": answer, "request_seconds": round(elapsed, 3)}

    run_case(results, "vision_png", vision_case)

    if args.video_path:

        def video_case() -> dict[str, Any]:
            response, elapsed = chat(
                base_url,
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": video_data_url(args.video_path)},
                            },
                            {
                                "type": "text",
                                "text": (
                                    "What single color fills this video? "
                                    "Reply with one word."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=16,
            )
            answer = content(response).strip()
            if "red" not in answer.lower():
                raise AssertionError(repr(answer))
            return {
                "answer": answer,
                "bytes": Path(args.video_path).stat().st_size,
                "request_seconds": round(elapsed, 3),
            }

        run_case(results, "video_mp4", video_case)

    if not args.skip_long:
        filler = args.filler_phrase * args.long_repetitions
        midpoint = len(filler) // 2
        context = (
            filler[:midpoint]
            + " The unique verification code is ORCHID. "
            + filler[midpoint:]
        )
        system_message = {
            "role": "system",
            "content": "Reference text follows. Preserve it for later lookup.\n" + context,
        }

        def long_retrieval_case() -> dict[str, Any]:
            response, elapsed = chat(
                base_url,
                [
                    system_message,
                    {
                        "role": "user",
                        "content": "What is the unique verification code? Reply with that word only.",
                    },
                ],
                max_tokens=16,
            )
            answer = content(response).strip()
            if answer != "ORCHID":
                raise AssertionError(repr(answer))
            usage = response.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            if prompt_tokens < args.min_long_tokens:
                raise AssertionError(
                    f"long probe produced only {prompt_tokens} prompt tokens; "
                    f"required {args.min_long_tokens}"
                )
            return {
                "answer": answer,
                "prompt_tokens": prompt_tokens,
                "request_seconds": round(elapsed, 3),
            }

        run_case(results, "long_retrieval", long_retrieval_case)

        def prefix_hit_case() -> dict[str, Any]:
            metric_name = "vllm:prefix_cache_hits_total"
            before = metric(base_url, metric_name)
            response, elapsed = chat(
                base_url,
                [
                    system_message,
                    {
                        "role": "user",
                        "content": "Repeat the unique verification code, and output no other text.",
                    },
                ],
                max_tokens=16,
            )
            after = metric(base_url, metric_name)
            answer = content(response).strip()
            hit_tokens = int(after - before)
            if answer != "ORCHID":
                raise AssertionError(repr(answer))
            if hit_tokens <= 0:
                raise AssertionError("no prefix-cache hit was observed")
            return {
                "answer": answer,
                "prefix_hit_tokens": hit_tokens,
                "request_seconds": round(elapsed, 3),
            }

        run_case(results, "prefix_hit", prefix_hit_case)

    failures = [name for name, result in results.items() if not result["ok"]]
    summary = {"ok": not failures, "failures": failures, "results": results}
    print(json.dumps(summary, sort_keys=True), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
