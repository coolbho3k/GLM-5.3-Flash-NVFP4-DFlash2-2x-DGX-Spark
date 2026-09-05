#!/usr/bin/env python3
"""Add separate async2 fat-GEMM symbols; preserve installed control sources.

Place build_exl3_fat_pipeline.py alongside this script (or on PYTHONPATH).
Both isolated probes and the native extension use its same transformation.
"""
from pathlib import Path
import re
import sys

from build_exl3_fat_pipeline import pipelined_source


def patch(root):
    root = Path(root)
    quant = root / 'quant'
    path = root / 'bindings.cpp'
    bindings = path.read_text()
    if 'glm53_fat_pipeline_version' in bindings:
        raise ValueError('Fat pipeline already installed; use a clean base image')
    original = (quant / 'exl3_fat_gemm.cu').read_text()
    header = (quant / 'exl3_fat_gemm.cuh').read_text()
    candidate = pipelined_source(original)
    for old, new in [('exl3_fat_gemm_scatter', 'exl3_fat_gemm_scatter_async2'),
                     ('exl3_fat_gemm', 'exl3_fat_gemm_async2')]:
        candidate = re.sub(r'\b' + old + r'\b', new, candidate)
        header = re.sub(r'\b' + old + r'\b', new, header)
    include = '#include "quant/exl3_fat_gemm.cuh"'
    binding = '    m.def("exl3_fat_gemm_scatter", &exl3_fat_gemm_scatter, "exl3_fat_gemm_scatter");'
    if bindings.count(include) != 1 or bindings.count(binding) != 1:
        raise ValueError('Expected existing E2 include and binding anchors')
    bindings = bindings.replace(include, include + '\n#include "quant/exl3_fat_gemm_async2.cuh"', 1)
    bindings = bindings.replace(binding, binding + '''
    m.def("exl3_fat_gemm_async2", &exl3_fat_gemm_async2);
    m.def("exl3_fat_gemm_scatter_async2", &exl3_fat_gemm_scatter_async2);
    m.def("glm53_fat_pipeline_version", []() { return 1; });''', 1)
    # Validate all transformations before touching files. The control CU/CUH
    # and all decode sources are deliberately left byte-for-byte unchanged.
    (quant / 'exl3_fat_gemm_async2.cu').write_text(candidate)
    (quant / 'exl3_fat_gemm_async2.cuh').write_text(header)
    path.write_text(bindings)


if __name__ == '__main__':
    patch(sys.argv[1])
