#!/usr/bin/env python3
"""Add separate async2 fat-GEMM symbols; preserve installed control sources.

Place build_exl3_fat_pipeline.py alongside this script (or on PYTHONPATH).
Both isolated probes and the native extension use its same transformation.
"""
from pathlib import Path
import re
import sys

from build_exl3_fat_pipeline import pipelined_source, small_tile_source


def patch(root):
    root = Path(root)
    quant = root / 'quant'
    path = root / 'bindings.cpp'
    bindings = path.read_text()
    if 'glm53_fat_pipeline_version' in bindings:
        raise ValueError('Fat pipeline already installed; use a clean base image')
    original = (quant / 'exl3_fat_gemm.cu').read_text()
    original_header = (quant / 'exl3_fat_gemm.cuh').read_text()
    generated = {}
    for suffix, candidate in [('async2', pipelined_source(original)),
                              ('async2_m64', small_tile_source(pipelined_source(original)))]:
        header = original_header
        for old in ['exl3_fat_gemm_scatter', 'exl3_fat_gemm']:
            candidate = re.sub(r'\b' + old + r'\b', old + '_' + suffix, candidate)
            header = re.sub(r'\b' + old + r'\b', old + '_' + suffix, header)
        generated[f'exl3_fat_gemm_{suffix}.cu'] = candidate
        generated[f'exl3_fat_gemm_{suffix}.cuh'] = header
    include = '#include "quant/exl3_fat_gemm.cuh"'
    binding = '    m.def("exl3_fat_gemm_scatter", &exl3_fat_gemm_scatter, "exl3_fat_gemm_scatter");'
    if bindings.count(include) != 1 or bindings.count(binding) != 1:
        raise ValueError('Expected existing E2 include and binding anchors')
    bindings = bindings.replace(include, include + '\n#include "quant/exl3_fat_gemm_async2.cuh"\n#include "quant/exl3_fat_gemm_async2_m64.cuh"', 1)
    bindings = bindings.replace(binding, binding + '''
    m.def("exl3_fat_gemm_async2", &exl3_fat_gemm_async2);
    m.def("exl3_fat_gemm_scatter_async2", &exl3_fat_gemm_scatter_async2);
    m.def("exl3_fat_gemm_async2_m64", &exl3_fat_gemm_async2_m64);
    m.def("exl3_fat_gemm_scatter_async2_m64", &exl3_fat_gemm_scatter_async2_m64);
    m.def("glm53_fat_pipeline_version", []() { return 2; });''', 1)
    # Validate all transformations before touching files. The control CU/CUH
    # and all decode sources are deliberately left byte-for-byte unchanged.
    for name, source in generated.items():
        (quant / name).write_text(source)
    path.write_text(bindings)


if __name__ == '__main__':
    patch(sys.argv[1])
