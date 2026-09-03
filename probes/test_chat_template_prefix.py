#!/usr/bin/env python3
"""Regression check for thinking-toggle prefix-cache stability."""

import json
from pathlib import Path

from jinja2 import Environment

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "chat_template_mm.jinja"


def render(enable_thinking: bool, tools: list[dict] | None = None) -> str:
    env = Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, **kwargs)
    template = env.from_string(TEMPLATE.read_text())
    return template.render(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow-up"},
        ],
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def main() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "status",
            "description": "Return status",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    for tools in (None, [tool]):
        thinking = render(True, tools)
        no_thinking = render(False, tools)
        assert "<|system|>Reasoning Effort: High" in thinking
        assert "<|system|>Reasoning Effort: High" in no_thinking
        assert no_thinking == thinking + "</think>", (
            thinking[-100:],
            no_thinking[-100:],
        )
    print("chat-template thinking toggle is prefix-stable")


if __name__ == "__main__":
    main()
