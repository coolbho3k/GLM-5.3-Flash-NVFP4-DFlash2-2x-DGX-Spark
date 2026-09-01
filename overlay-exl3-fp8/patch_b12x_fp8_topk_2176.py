#!/usr/bin/env python3
"""Enable GLM zero-RoPE FP8 prefill for vLLM's padded top-k 2176 shape."""

from pathlib import Path


PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "b12x/attention/_shared/mla/prefill.py"
)


def replace(old: str, new: str) -> None:
    source = PATH.read_text()
    if new in source:
        return
    if source.count(old) != 1:
        raise RuntimeError(
            f"B12X FP8 top-k patch anchor count is {source.count(old)}, expected 1"
        )
    PATH.write_text(source.replace(old, new, 1))


replace(
    "if _mg_glm and topk in (512, 1024, 2048):",
    "if _mg_glm and topk in (512, 1024, 2048, 2176):",
)
replace(
    '"GLM_NSA topk in {512, 1024, 2048}; "',
    '"GLM-family FP8 topk in {512, 1024, 2048, 2176}; "',
)
print(f"B12X GLM zero-RoPE FP8 top-k 2176 enabled in {PATH}")
