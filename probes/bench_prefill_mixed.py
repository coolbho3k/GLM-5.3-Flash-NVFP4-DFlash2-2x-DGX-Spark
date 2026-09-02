#!/usr/bin/env python3
"""Deterministic pure-prefill and mixed prefill/decode serving benchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PHRASES = (
    "amber cedar silver mountain ",
    "saffron nebula quartz meadow ",
    "violet tundra copper harbor ",
    "indigo maple granite valley ",
)


def health(base_url: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url}/health", timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"health returned {response.status}")
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=30) as response:
        return json.load(response)


def stream_chat(
    base_url: str,
    payload: dict[str, Any],
    *,
    on_first_token: Callable[[], None] | None = None,
    timeout: float = 1800,
) -> dict[str, Any]:
    request_payload = dict(payload)
    request_payload["stream"] = True
    request_payload["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(request_payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    event_times: list[float] = []
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or choices[0].get("text")
                    or ""
                )
                if piece:
                    now = time.perf_counter()
                    if not event_times and on_first_token is not None:
                        on_first_token()
                    event_times.append(now)
                    pieces.append(str(piece))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    finished = time.perf_counter()
    if not event_times:
        raise RuntimeError("stream returned no output token")
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    if completion_tokens <= 0 or prompt_tokens <= 0:
        raise RuntimeError(f"stream did not report complete usage: {usage}")
    gaps = [
        later - earlier for earlier, later in zip(event_times, event_times[1:])
    ]
    sorted_gaps = sorted(gaps)
    p95_gap = (
        sorted_gaps[min(len(sorted_gaps) - 1, math.ceil(0.95 * len(sorted_gaps)) - 1)]
        if sorted_gaps
        else 0.0
    )
    ttft = event_times[0] - started
    decode_seconds = max(finished - event_times[0], 1e-9)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_seconds": round(ttft, 6),
        "elapsed_seconds": round(finished - started, 6),
        "decode_seconds": round(decode_seconds, 6),
        "decode_tokens_per_second": round(completion_tokens / decode_seconds, 6),
        "mean_stream_gap_seconds": round(statistics.mean(gaps), 6) if gaps else 0.0,
        "p95_stream_gap_seconds": round(p95_gap, 6),
        "max_stream_gap_seconds": round(max(gaps), 6) if gaps else 0.0,
        "stream_events": len(event_times),
        "content_preview": "".join(pieces)[:160],
    }


def prefill_messages(
    target_tokens: int,
    *,
    case_id: str,
    corpus_seed: str,
    phrase_index: int,
) -> list[dict[str, str]]:
    # These phrases are approximately six GLM tokens per repetition. The exact
    # prompt-token count comes from server usage and is what all rates use.
    repetitions = max(1, (target_tokens - 64) // 6)
    phrase = PHRASES[phrase_index % len(PHRASES)]
    marker = f"B12X A/B corpus={corpus_seed} case={case_id}. "
    reference = marker + phrase * repetitions
    return [
        {
            "role": "system",
            "content": "Read this synthetic reference before answering.\n" + reference,
        },
        {"role": "user", "content": "Reply with exactly OK."},
    ]


def prefill_once(
    base_url: str,
    target_tokens: int,
    *,
    case_id: str,
    corpus_seed: str,
    phrase_index: int,
) -> dict[str, Any]:
    result = stream_chat(
        base_url,
        {
            "model": "glm-5.3-flash",
            "messages": prefill_messages(
                target_tokens,
                case_id=case_id,
                corpus_seed=corpus_seed,
                phrase_index=phrase_index,
            ),
            "temperature": 0,
            "max_tokens": 4,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    result["target_prompt_tokens"] = target_tokens
    result["input_tokens_per_second"] = round(
        result["prompt_tokens"] / result["ttft_seconds"], 6
    )
    return result


def decode_payload(tokens: int, *, case_id: str, corpus_seed: str) -> dict[str, Any]:
    return {
        "model": "glm-5.3-flash",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"B12X A/B corpus={corpus_seed} case={case_id}. "
                    "Write an indefinitely continuing numbered list of short, "
                    "independent software-testing observations."
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": tokens,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def mixed_wave(
    base_url: str,
    *,
    round_index: int,
    decode_tokens: int,
    prefill_tokens: int,
    corpus_seed: str,
) -> dict[str, Any]:
    control = stream_chat(
        base_url,
        decode_payload(
            decode_tokens,
            case_id=f"control-{round_index}",
            corpus_seed=corpus_seed,
        ),
    )

    first_token = threading.Event()
    decode_box: dict[str, Any] = {}

    def run_decode() -> None:
        try:
            decode_box["result"] = stream_chat(
                base_url,
                decode_payload(
                    decode_tokens,
                    case_id=f"mixed-decode-{round_index}",
                    corpus_seed=corpus_seed,
                ),
                on_first_token=first_token.set,
            )
        except BaseException as exc:
            decode_box["error"] = exc
            first_token.set()

    thread = threading.Thread(target=run_decode, daemon=True)
    thread.start()
    if not first_token.wait(timeout=120):
        raise RuntimeError("mixed decoder did not emit its first token")
    if "error" in decode_box:
        raise decode_box["error"]

    concurrent_prefill = prefill_once(
        base_url,
        prefill_tokens,
        case_id=f"mixed-prefill-{round_index}",
        corpus_seed=corpus_seed,
        phrase_index=round_index + 2,
    )
    thread.join(timeout=1800)
    if thread.is_alive():
        raise RuntimeError("mixed decoder did not finish")
    if "error" in decode_box:
        raise decode_box["error"]
    mixed_decode = decode_box["result"]

    return {
        "round": round_index,
        "control_decode": control,
        "mixed_decode": mixed_decode,
        "concurrent_prefill": concurrent_prefill,
        "mixed_over_control_decode_tps": round(
            mixed_decode["decode_tokens_per_second"]
            / control["decode_tokens_per_second"],
            6,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus-seed", default="glm53-b12x-w4a16-ab-v1")
    parser.add_argument("--pure-targets", default="8192,32768,131072")
    parser.add_argument("--mixed-rounds", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=384)
    parser.add_argument("--mixed-prefill-tokens", type=int, default=32768)
    parser.add_argument("--skip-pure", action="store_true")
    parser.add_argument("--skip-mixed", action="store_true")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    models = health(base_url)
    # Warm kernels and the streaming API without sharing any benchmark prefix.
    stream_chat(
        base_url,
        {
            "model": "glm-5.3-flash",
            "messages": [{"role": "user", "content": "Warmup: list test numbers."}],
            "temperature": 0,
            "max_tokens": 32,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    report: dict[str, Any] = {
        "schema": 1,
        "label": args.label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "url": base_url,
        "corpus_seed": args.corpus_seed,
        "models": models,
        "pure_prefill": [],
        "mixed": [],
    }

    if not args.skip_pure:
        targets = [int(item) for item in args.pure_targets.split(",") if item]
        for index, target in enumerate(targets):
            result = prefill_once(
                base_url,
                target,
                case_id=f"pure-{index}-{target}",
                corpus_seed=args.corpus_seed,
                phrase_index=index,
            )
            report["pure_prefill"].append(result)
            print(json.dumps({"pure_prefill": result}, sort_keys=True), flush=True)

    if not args.skip_mixed:
        for round_index in range(args.mixed_rounds):
            result = mixed_wave(
                base_url,
                round_index=round_index,
                decode_tokens=args.decode_tokens,
                prefill_tokens=args.mixed_prefill_tokens,
                corpus_seed=args.corpus_seed,
            )
            report["mixed"].append(result)
            print(json.dumps({"mixed": result}, sort_keys=True), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
