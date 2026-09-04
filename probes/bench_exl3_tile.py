#!/usr/bin/env python3
"""Isolated EXL3 K4 N128/N256 screen using already-compiled kernel variants.

Changes only this probe process's dispatch table, with ABI/pointer checks and
restoration in finally. Run with the server stopped; do not rely on Linux's
available-memory estimate for a second CUDA context. Never inject this into a
live worker. Captures a fresh
graph for each choice. Synthetic experts are not a model-quality benchmark.
"""
import argparse
import ctypes
import json
import statistics
from types import SimpleNamespace

import torch
import exllamav3_ext as ext
from vllm.model_executor.layers.quantization.exl3 import (
    apply_exl3_fused_moe, build_exl3_fused_state,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experts', type=int, default=288)
    parser.add_argument('--rows', default='8,48,128')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.035)
    torch.manual_seed(42)
    library = ctypes.CDLL(ext.__file__)
    table = (ctypes.c_void_p * 18).in_dll(library, 'exl3_moe_kernel_instances')
    expected = []
    for tile in (128, 256):
        getter = getattr(library, f'_Z23exl3_moe_kernel_k4_n{tile}v')
        getter.restype = ctypes.c_void_p
        getter.argtypes = []
        expected.append(getter())
    assert all(expected) and expected[0] != expected[1]
    assert list(table[8:10]) == expected, 'Unexpected extension dispatch-table ABI'
    hidden, intermediate = 4096, 1024
    experts = []
    for _ in range(args.experts):
        pack = {}
        for name, k, n in [('gate', hidden, intermediate), ('up', hidden, intermediate),
                           ('down', intermediate, hidden)]:
            pack[name] = SimpleNamespace(
                trellis=torch.randint(-32768, 32767, (k // 16, n // 16, 64),
                                      dtype=torch.int16, device='cuda'),
                suh=torch.ones(k, dtype=torch.float16, device='cuda'),
                svh=torch.full((n,), 0.02, dtype=torch.float16, device='cuda'))
        experts.append(pack)
    layer = SimpleNamespace(w13_trellis=torch.empty(0, device='cuda'),
                            _exl3_hidden_size=hidden, _exl3_intermediate_local=intermediate,
                            _exl3_bits=4)
    build_exl3_fused_state(layer, experts)
    try:
        with torch.inference_mode():
            for rows in map(int, args.rows.split(',')):
                x = torch.randn(rows, hidden, dtype=torch.bfloat16, device='cuda')
                ids = torch.rand(rows, args.experts, device='cuda').topk(8, dim=-1).indices
                weights = torch.full((rows, 8), 0.125, device='cuda')

                def run():
                    return apply_exl3_fused_moe(x, ids, weights, layer, experts, None, 10.0)

                result = {'rows': rows, 'experts': args.experts, 'variants': {}}
                baseline = None
                for label, pointer in [('n256', expected[1]), ('n128', expected[0]),
                                       ('n256_repeat', expected[1])]:
                    torch.cuda.synchronize()
                    table[9] = pointer
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            run()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        out = run()
                    times = []
                    for _ in range(3):
                        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        start.record()
                        for _ in range(20):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        times.append(start.elapsed_time(end) / 20)
                    assert out.isfinite().all()
                    if baseline is None:
                        baseline = out.clone()
                    delta = out - baseline
                    result['variants'][label] = {
                        'ms': statistics.median(times),
                        'relative_rmse': (delta.square().mean() / baseline.square().mean()).sqrt().item(),
                        'max_abs_difference': delta.abs().max().item()}
                    del graph, out
                print(json.dumps(result), flush=True)
    finally:
        torch.cuda.synchronize()
        table[9] = expected[1]


if __name__ == '__main__':
    main()
