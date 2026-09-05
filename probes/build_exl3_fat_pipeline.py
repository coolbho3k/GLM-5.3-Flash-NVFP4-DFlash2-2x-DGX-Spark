#!/usr/bin/env python3
"""Build isolated current and double-buffered E2 kernels without Torch/CUDA use.

nvcc compiles for SM121, but this builder never creates a GPU context. It uses
the installed EXL3 headers and the recipe's fat-kernel source. Serving files
are not modified. Numerical tests are separate and require the server stopped.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def pipelined_source(source):
    def once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f'Expected one E2 pipeline anchor: {old!r}')
        source = source.replace(old, new, 1)

    once('''    half* sh_a = reinterpret_cast<half*>(shared_raw);
    uint16_t* sh_b = reinterpret_cast<uint16_t*>(sh_a + FAT_TILE_M * FAT_TILE_K);
    float* sh_c = reinterpret_cast<float*>(sh_b + FAT_N_BLOCKS * FAT_PACKED_WORDS);''', '''    half* ring_a = reinterpret_cast<half*>(shared_raw);
    uint16_t* ring_b = reinterpret_cast<uint16_t*>(ring_a + 2 * FAT_TILE_M * FAT_TILE_K);
    float* sh_c = reinterpret_cast<float*>(ring_b + 2 * FAT_N_BLOCKS * FAT_PACKED_WORDS);''')
    start = source.index('    for (int k_block = 0; k_block < size_k / FAT_TILE_K; ++k_block)')
    end = source.index('        FragB frag_b0;', start)
    old = source[start:end]
    if old.count('__syncthreads();') != 1 or old.count('b_src[t]') != 1:
        raise ValueError('Unexpected baseline load loop')
    source = source[:start] + '''    auto prefetch = [&](int stage, int k_block) {
        half* dst_a = ring_a + stage * FAT_TILE_M * FAT_TILE_K;
        uint16_t* dst_b = ring_b + stage * FAT_N_BLOCKS * FAT_PACKED_WORDS;
        int a_row = t / 2;
        int a_col8 = t & 1;
        int a_dst_col8 = a_col8 ^ ((a_row >> 2) & 1);
        int4* a_dst = reinterpret_cast<int4*>(dst_a) + a_row * 2 + a_dst_col8;
        if (m_base + a_row < size_m) {
            const int4* a_src = reinterpret_cast<const int4*>(
                a + (m_base + a_row) * size_k + k_block * FAT_TILE_K);
            cp_async(a_dst, a_src + a_col8);
        } else {
            // Match the original zero-filled M tail without forming an
            // out-of-range global address or issuing an invalid async copy.
            *a_dst = int4{};
        }
        if (t < 64) {
            const int4* b_src = reinterpret_cast<const int4*>(
                packed + (k_block * tiles_n + n_base / 16) * FAT_PACKED_WORDS);
            cp_async(reinterpret_cast<int4*>(dst_b) + t, b_src + t);
        }
        cp_async_fence();
    };
    prefetch(0, 0);
    for (int k_block = 0; k_block < size_k / FAT_TILE_K; ++k_block)
    {
        cp_async_wait<0>();
        __syncthreads();
        half* sh_a = ring_a + (k_block & 1) * FAT_TILE_M * FAT_TILE_K;
        uint16_t* sh_b = ring_b + (k_block & 1) * FAT_N_BLOCKS * FAT_PACKED_WORDS;
        if (k_block + 1 < size_k / FAT_TILE_K)
            prefetch((k_block + 1) & 1, k_block + 1);

''' + source[end:]
    once('''    size_t shared = FAT_TILE_M * FAT_TILE_K * sizeof(half)
                  + FAT_N_BLOCKS * FAT_PACKED_WORDS * sizeof(uint16_t)''',
         '''    size_t shared = 2 * FAT_TILE_M * FAT_TILE_K * sizeof(half)
                  + 2 * FAT_N_BLOCKS * FAT_PACKED_WORDS * sizeof(uint16_t)''')
    return source


def small_tile_source(source):
    """Separate, not yet GPU-qualified M64 experiment over the async2 source."""
    anchor = 'constexpr int FAT_TILE_M = 128;'
    if source.count(anchor) != 1 or 'auto prefetch = ' not in source:
        raise ValueError('M64 probe requires the original-size async2 source')
    source = source.replace(anchor, 'constexpr int FAT_TILE_M = 64;', 1)
    start = source.index('        int4* a_dst = reinterpret_cast<int4*>(dst_a)')
    end = source.index('        if (t < 64) {', start)
    # Only the first 128 threads copy the 64 rows; all 256 threads still
    # participate in B loads, barriers and the N-direction MMA work.
    source = source[:start] + '        if (a_row < FAT_TILE_M) {\n' + source[start:end] + '        }\n' + source[end:]
    return source


def standalone(source, stages):
    body = source[source.index('namespace {'):source.index('\nvoid check_common(')]
    headers = '''#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "quant/exl3_dq.cuh"
#include "quant/hadamard_inner.cuh"
'''
    wrapper = '''
} // namespace
constexpr int SHARED_BYTES = STAGES * (FAT_TILE_M * FAT_TILE_K * 2 + 8 * 64 * 2) + 16 * 128 * 4;
extern "C" int fat_probe_shared_bytes() { return SHARED_BYTES; }
extern "C" int fat_probe_tile_m() { return FAT_TILE_M; }
extern "C" int fat_probe_occupancy(int scatter) {
    int blocks = 0;
    auto kernel = scatter ? exl3_fat_gemm_kernel<true> : exl3_fat_gemm_kernel<false>;
    auto rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 256, SHARED_BYTES);
    return rc == cudaSuccess ? blocks : -static_cast<int>(rc);
}
extern "C" int fat_probe_launch(void** args, int m, int k, int n, int scatter, void* stream) {
    if (m <= 0 || m > 8192 || k <= 0 || k > 4096 || k % 16 ||
        n <= 0 || n > 4096 || n % 128 || (scatter != 0 && scatter != 1))
        return cudaErrorInvalidValue;
    auto kernel = scatter ? exl3_fat_gemm_kernel<true> : exl3_fat_gemm_kernel<false>;
    return cudaLaunchKernel((const void*)kernel, dim3(n / 128, (m + FAT_TILE_M - 1) / FAT_TILE_M),
                           dim3(256), args, SHARED_BYTES, (cudaStream_t)stream);
}
'''.replace('STAGES', str(stages))
    return headers + body + wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--extra-tile64', action='store_true',
                        help='Also stage the separately unqualified smaller-M-tile candidate')
    args = parser.parse_args()
    root = Path('/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext')
    output = args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError('Never build into the installed extension')
    output.mkdir(parents=True, exist_ok=True)
    original = args.source.read_text()
    variants = [('control', original, 1), ('async2', pipelined_source(original), 2)]
    if args.extra_tile64:
        variants.append(('async2_m64', small_tile_source(pipelined_source(original)), 2))
    for label, source, stages in variants:
        generated = standalone(source, stages)
        path = output / (label + '.cu')
        path.write_text(generated)
        cmd = ['/usr/local/cuda/bin/nvcc', '-O3', '--use_fast_math', '-std=c++17',
               '-arch=sm_121', '--shared', '-Xcompiler=-fPIC', '-Xlinker=-Bsymbolic',
               '-Xptxas=-v', '-lineinfo', '-I', str(root), str(path),
               '-o', str(output / (label + '.so'))]
        subprocess.run(cmd, check=True)
        print(json.dumps(dict(variant=label, command=cmd,
              original_sha256=hashlib.sha256(original.encode()).hexdigest(),
              generated_sha256=hashlib.sha256(generated.encode()).hexdigest())), flush=True)


if __name__ == '__main__':
    main()
