#!/usr/bin/env python3
"""Validate GLM-5.3 reasoning-effort compatibility and parser behavior."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

ALL_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
EXPECTED_NATIVE = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}
_UNSET = object()


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {body}") from exc


def tokenize(
    base_url: str,
    model: str,
    timeout: float,
    effort: object = _UNSET,
    *,
    legacy_off: bool = False,
) -> tuple[int, ...]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply OK."}],
        "add_generation_prompt": True,
    }
    chat_template_kwargs: dict[str, Any] = {}
    if effort is not _UNSET:
        chat_template_kwargs["reasoning_effort"] = effort
    if legacy_off:
        chat_template_kwargs["enable_thinking"] = False
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    body = post_json(f"{base_url}/tokenize", payload, timeout)
    return tuple(body["tokens"])


def chat_smoke(
    base_url: str, model: str, timeout: float, effort: object = _UNSET
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Return exactly OK."}],
        "max_tokens": 48,
        "temperature": 0,
    }
    if effort is not _UNSET:
        payload["reasoning_effort"] = effort
    return post_json(f"{base_url}/v1/chat/completions", payload, timeout)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    signatures: dict[str, tuple[int, ...]] = {
        "default": tokenize(base_url, args.model, args.timeout)
    }
    for effort in ALL_EFFORTS:
        signatures[effort] = tokenize(
            base_url, args.model, args.timeout, effort
        )
    signatures["legacy_off"] = tokenize(
        base_url, args.model, args.timeout, legacy_off=True
    )

    failures: list[str] = []
    aliases = {
        "default": "high",
        "minimal": "low",
        "medium": "high",
        "xhigh": "max",
        "legacy_off": "none",
    }
    for alias, target in aliases.items():
        if signatures[alias] != signatures[target]:
            failures.append(f"{alias} did not render identically to {target}")

    native_signatures = {
        name: signatures[name] for name in ("none", "low", "high", "max")
    }
    if len(set(native_signatures.values())) != len(native_signatures):
        failures.append("none/low/high/max did not render as four distinct prompts")

    generation: dict[str, Any] = {}
    if not args.skip_generation:
        generation_cases = [("default", _UNSET)] + [
            (effort, effort) for effort in ALL_EFFORTS
        ]
        for label, effort in generation_cases:
            body = chat_smoke(base_url, args.model, args.timeout, effort)
            message = body["choices"][0]["message"]
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            content = message.get("content")
            generation[label] = {
                "finish_reason": body["choices"][0].get("finish_reason"),
                "reasoning_chars": len(reasoning or ""),
                "content_chars": len(content or ""),
            }
            if label == "none" and reasoning:
                failures.append("reasoning_effort=none returned parsed reasoning")
            if label != "none" and not reasoning:
                failures.append(f"{label} returned no parsed reasoning")
            if label == "none" and not content:
                failures.append("reasoning_effort=none returned no content")

        responses = post_json(
            f"{base_url}/v1/responses",
            {
                "model": args.model,
                "input": "Return exactly OK.",
                "reasoning": {"effort": "medium"},
                "max_output_tokens": 24,
            },
            args.timeout,
        )
        reasoning_items = [
            item for item in responses.get("output", []) if item.get("type") == "reasoning"
        ]
        generation["responses_medium"] = {
            "status": responses.get("status"),
            "reasoning_items": len(reasoning_items),
        }
        if not reasoning_items:
            failures.append("Responses API medium request returned no reasoning item")

    report = {
        "ok": not failures,
        "base_url": base_url,
        "model": args.model,
        "default_native_effort": "high",
        "compatibility_map": EXPECTED_NATIVE,
        "token_counts": {name: len(tokens) for name, tokens in signatures.items()},
        "generation": generation,
        "failures": failures,
    }
    if failures:
        raise AssertionError(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://10.100.32.1:8000")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    try:
        report = validate(args)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
