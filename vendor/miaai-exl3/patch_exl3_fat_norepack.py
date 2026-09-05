#!/usr/bin/env python3
"""Add a separate-buffer M64 gate/up entry point; retain all existing kernels."""
from pathlib import Path
import sys

from build_exl3_fat_pipeline import pipelined_source, small_tile_source
from build_exl3_fat_norepack import separate_gate_up_source


HEADER = '''#pragma once
#include <torch/extension.h>
void exl3_fat_gate_up_norepack(at::Tensor a, at::Tensor gate, at::Tensor up,
    at::Tensor out, at::Tensor sg, at::Tensor su, int64_t K, bool mcg, bool mul1);
'''


def native_source(original):
    candidate = separate_gate_up_source(small_tile_source(pipelined_source(original)))
    marker = 'template <bool scatter>\nvoid launch('
    if candidate.count(marker) != 1:
        raise ValueError('Unexpected native launch wrapper')
    candidate = candidate[:candidate.index(marker)]
    candidate = candidate.replace('#include "exl3_fat_gemm.cuh"', '#include "exl3_fat_norepack.cuh"')
    return candidate + '''} // namespace

void exl3_fat_gate_up_norepack(at::Tensor a, at::Tensor gate, at::Tensor up,
    at::Tensor out, at::Tensor sg, at::Tensor su, int64_t K, bool mcg, bool mul1)
{
    check_common(a, gate, out, sg, K, mcg, mul1);
    check_common(a, up, out, su, K, mcg, mul1);
    TORCH_CHECK(gate.sizes() == up.sizes() && sg.numel() == su.numel(),
                "gate/up dimensions must match");
    TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == 2 * sg.numel(),
                "out must be [M, gate_N + up_N]");
    TORCH_CHECK(a.size(0) > 0 && a.size(0) <= 8192 &&
                a.size(1) > 0 && a.size(1) <= 4096 &&
                sg.numel() > 0 && 2 * sg.numel() <= 4096,
                "no-repack kernel dimensions exceed validated bounds");
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int m = static_cast<int>(a.size(0));
    int k = static_cast<int>(a.size(1));
    int n = static_cast<int>(2 * sg.numel());
    size_t shared = 2 * (FAT_TILE_M * FAT_TILE_K * sizeof(half)
                  + FAT_N_BLOCKS * FAT_PACKED_WORDS * sizeof(uint16_t))
                  + 16 * FAT_TILE_N * sizeof(float);
    exl3_fat_gemm_kernel<false><<<dim3(n / FAT_TILE_N, (m + FAT_TILE_M - 1) / FAT_TILE_M),
        dim3(FAT_THREADS), shared, stream>>>(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const uint16_t*>(gate.data_ptr()),
        reinterpret_cast<const uint16_t*>(up.data_ptr()),
        reinterpret_cast<float*>(out.data_ptr()),
        reinterpret_cast<const half*>(sg.data_ptr()),
        reinterpret_cast<const half*>(su.data_ptr()),
        nullptr, nullptr, m, k, n);
    cuda_check(cudaPeekAtLastError());
}
'''


def patch(root):
    root = Path(root)
    quant = root / 'quant'
    bindings_path = root / 'bindings.cpp'
    bindings = bindings_path.read_text()
    if 'glm53_fat_norepack_version' in bindings:
        raise ValueError('No-repack already installed; use a clean M64 base')
    include = '#include "quant/exl3_fat_gemm_async2_m64.cuh"'
    binding = '    m.def("glm53_fat_pipeline_version", []() { return 2; });'
    if bindings.count(include) != 1 or bindings.count(binding) != 1:
        raise ValueError('Requires additive fat pipeline v2')
    candidate = native_source((quant / 'exl3_fat_gemm.cu').read_text())
    bindings = bindings.replace(include, include + '\n#include "quant/exl3_fat_norepack.cuh"', 1)
    bindings = bindings.replace(binding, binding + '''
    m.def("exl3_fat_gate_up_norepack", &exl3_fat_gate_up_norepack);
    m.def("glm53_fat_norepack_version", []() { return 1; });''', 1)
    (quant / 'exl3_fat_norepack.cu').write_text(candidate)
    (quant / 'exl3_fat_norepack.cuh').write_text(HEADER)
    bindings_path.write_text(bindings)


if __name__ == '__main__':
    patch(sys.argv[1])
