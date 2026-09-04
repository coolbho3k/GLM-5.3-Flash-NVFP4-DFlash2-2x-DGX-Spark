#!/usr/bin/env python3
"""Rebuild the installed, already-patched native sources inside an image.

Used by the fast experiment overlay. Full recipe builds patch sources before
their normal package build instead. Does not allocate a GPU.
"""
import importlib.util
from pathlib import Path
import shutil
import tempfile

from torch.utils.cpp_extension import load


def main():
    spec = importlib.util.find_spec('exllamav3_ext')
    assert spec is not None and spec.origin
    destination = Path(spec.origin)
    source = destination.parent / 'exllamav3/exllamav3_ext'
    sources = sorted(str(p) for p in source.rglob('*') if p.suffix in ('.c', '.cpp', '.cu'))
    assert sources and (source / 'quant/comp_units/glm53_exl3_moe_fast.cu').is_file()
    cuda_headers = destination.parent / 'nvidia/cu13/include'
    assert (cuda_headers / 'cusparse.h').is_file(), 'Missing packaged CUDA headers'
    with tempfile.TemporaryDirectory(prefix='glm53-exl3-native-') as build:
        module = load(name='exllamav3_ext', sources=sources,
                      extra_include_paths=[str(source)], build_directory=build,
                      # CUDA's own runtime headers must precede wheel library
                      # headers: nvcc and wheel CUDA minor versions can differ.
                      extra_cflags=['-Ofast', '-isystem', str(cuda_headers)],
                      extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo',
                                         '-isystem', str(cuda_headers),
                                         '-Xcudafe', '--diag_suppress=177',
                                         '-Xcudafe', '--diag_suppress=20012'], verbose=True)
        assert module.glm53_fast_moe_version() == 1
        assert hasattr(module, 'exl3_fat_gemm_scatter')
        shutil.copyfile(module.__file__, destination)
        print(f'Installed rebuilt native extension: {destination}')


if __name__ == '__main__':
    main()
