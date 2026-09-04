#!/usr/bin/env python3
"""Register the pinned B12X MXFP8 linear kernel ahead of generic fallbacks."""

from pathlib import Path


PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/kernels/linear/__init__.py"
)


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one registry anchor, found {count}: {old!r}")
    return source.replace(old, new, 1)


source = PATH.read_text()
import_line = (
    "from vllm.model_executor.kernels.linear.mxfp8.b12x import (\n"
    "    B12xMxfp8LinearKernel,\n"
    ")\n"
)
if import_line not in source:
    anchor = (
        "from vllm.model_executor.kernels.linear.mxfp8.emulation import (\n"
        "    EmulationMxfp8LinearKernel,\n"
        ")\n"
    )
    source = replace_once(source, anchor, import_line + anchor)

kernel_entry = "        B12xMxfp8LinearKernel,\n"
if kernel_entry not in source:
    anchor = (
        "_POSSIBLE_MXFP8_KERNELS: dict[PlatformEnum, "
        "list[type[Mxfp8LinearKernel]]] = {\n"
        "    PlatformEnum.CUDA: [\n"
    )
    source = replace_once(source, anchor, anchor + kernel_entry)

PATH.write_text(source)
