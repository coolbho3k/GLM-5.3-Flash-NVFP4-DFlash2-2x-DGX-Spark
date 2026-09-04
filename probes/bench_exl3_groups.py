#!/usr/bin/env python3
"""Bounded, isolated EXL3 expert-group/transform test. Server MUST be stopped.

Synthetic weights and routing screen speed/correctness, not model quality.
The same precomputed inputs feed stock and candidate kernels. Timing includes
output clearing but excludes routing. Each variant has independent scratch and
locks; CUDA graphs never share an in-flight scratch allocation.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
import exllamav3_ext as ext
from vllm.model_executor.layers.quantization.exl3 import (
    build_exl3_fused_state, _exl3_moe_launch,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--libraries', type=Path, required=True)
    p.add_argument('--variants', default='g8,g4')
    p.add_argument('--rows', default='8,48,128')
    p.add_argument('--routing', default='uniform,correlated')
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--random-scales', action='store_true')
    p.add_argument('--input-scale', type=float, default=1.0)
    p.add_argument('--alias-shared-scales', action='store_true')
    p.add_argument('--independent-up-scales', action='store_true')
    args = p.parse_args()
    assert not (args.alias_shared_scales and args.independent_up_scales)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.05)
    torch.manual_seed(42)
    hidden, intermediate, n_experts, topk, cap = 4096, 1024, 288, 8, 128
    experts = []
    for _ in range(n_experts):
        pack = {}
        for name, k, n in [('gate', hidden, intermediate), ('up', hidden, intermediate),
                           ('down', intermediate, hidden)]:
            pack[name] = SimpleNamespace(
                trellis=torch.randint(-32768, 32767, (k // 16, n // 16, 64),
                                      dtype=torch.int16, device='cuda'),
                suh=torch.ones(k, dtype=torch.float16, device='cuda'),
                svh=torch.full((n,), 0.02, dtype=torch.float16, device='cuda'))
        experts.append(pack)
        if args.random_scales:
            for name in ('gate', 'up', 'down'):
                pack[name].suh.uniform_(0.25, 1.75)
                pack[name].svh.uniform_(0.005, 0.035)
            if not args.independent_up_scales:
                pack['up'].suh.copy_(pack['gate'].suh)
        if args.alias_shared_scales:
            pack['up'].suh = pack['gate'].suh
    layer = SimpleNamespace(w13_trellis=torch.empty(0, device='cuda'),
                            _exl3_hidden_size=hidden, _exl3_intermediate_local=intermediate,
                            _exl3_bits=4, _exl3_shared_w13_suh=args.alias_shared_scales)
    build_exl3_fused_state(layer, experts)
    variants = {}
    launch_metadata = {}
    for label in args.variants.split(','):
        group = int(label.split('_')[0][1:])
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        assert sms % group == 0
        concurrency = sms // group
        library = C.CDLL(str((args.libraries / f'{label}.so').resolve()))
        library.probe_init.restype = C.c_int
        library.probe_launch.restype = C.c_int
        library.probe_launch.argtypes = [C.POINTER(C.c_void_p), C.c_int, C.c_void_p]
        if label.endswith('_private'):
            assert args.alias_shared_scales, 'Private output requires verified shared input'
            library.probe_launch_private.restype = C.c_int
            library.probe_launch_private.argtypes = [C.POINTER(C.c_void_p), C.c_int,
                                                    C.c_int, C.c_void_p]
        init_rc = library.probe_init()
        assert init_rc == 0, ('unsafe or unsupported launch', label, init_rc)
        if hasattr(library, 'probe_concurrency'):
            concurrency = library.probe_concurrency()
            assert concurrency > 0
            launch_metadata[label] = {
                'concurrency': concurrency, 'grid_blocks': concurrency * group,
                'active_blocks_per_sm': library.probe_active_blocks_per_sm(),
                'dynamic_smem_bytes': library.probe_smem_bytes(),
            }
        scratch = [torch.empty((concurrency, cap, dim), dtype=torch.float16, device='cuda')
                   for dim in (hidden, hidden, intermediate, intermediate)]
        locks = torch.zeros(1024 * 1024 + 2 * 1024, dtype=torch.int32, device='cuda')
        variants[label] = (library, concurrency, scratch, locks)

    with torch.inference_mode():
        for rows in map(int, args.rows.split(',')):
            assert 0 < rows <= cap
            for routing in args.routing.split(','):
                x = torch.randn(rows, hidden, dtype=torch.float16, device='cuda') * args.input_scale
                if routing == 'uniform':
                    ids = torch.rand(rows, n_experts, device='cuda').topk(topk, dim=-1).indices
                elif routing == 'correlated':
                    # Each speculative block shares 6 of its 8 experts, with
                    # two token-specific experts from a disjoint pool.
                    base = torch.rand((rows + 7) // 8, n_experts // 2, device='cuda').topk(6, dim=-1).indices
                    common = base.repeat_interleave(8, dim=0)[:rows]
                    tail = torch.rand(rows, n_experts // 2, device='cuda').topk(2, dim=-1).indices + n_experts // 2
                    ids = torch.cat((common, tail), dim=1)
                else:
                    raise ValueError(routing)
                order = ids.flatten().argsort(stable=True)
                counts = torch.bincount(ids.flatten(), minlength=n_experts + 1)
                # Outside timing. Traffic estimate excludes scales/activations
                # and assumes every expert's packed weights stream per M16 tile.
                active_experts = int((counts > 0).sum().item())
                row_tiles = int(((counts + 15) // 16).sum().item())
                packed_weight_bytes = row_tiles * 3 * hidden * intermediate // 2
                tokens = torch.arange(rows, device='cuda').repeat_interleave(topk)[order]
                weights = torch.full((rows * topk,), 1 / topk, dtype=torch.float16, device='cuda')
                out = torch.empty(rows, hidden, dtype=torch.float32, device='cuda')
                result = {'rows': rows, 'routing': routing,
                          'active_experts': active_experts, 'expert_m16_tiles': row_tiles,
                          'estimated_packed_weight_bytes': packed_weight_bytes,
                          'random_scales': args.random_scales, 'input_scale': args.input_scale,
                          'native_fast': os.environ.get('GLM53_EXL3_MOE_FAST', '0'),
                          'aliased_scales': args.alias_shared_scales,
                          'independent_up_scales': args.independent_up_scales,
                          'launch_metadata': launch_metadata,
                          'variants': {}}
                baseline = None
                for label in ['stock', *variants, 'stock_repeat']:
                    values = None
                    if label.startswith('stock'):
                        def run():
                            out.zero_()
                            _exl3_moe_launch(ext.exl3_moe, x, out, counts, tokens, weights,
                                            layer._exl3_fused_temps, layer._exl3_ptrs, 4, 10.0, None)
                    else:
                        lib, concurrency, scratch, locks = variants[label]
                        values = [C.c_void_p(t.data_ptr()) for t in
                                  [x, *scratch, out, *layer._exl3_ptrs.values(), counts, tokens, weights]]
                        values += [C.c_int(v) for v in [hidden, intermediate, n_experts, topk, cap, concurrency]]
                        values += [C.c_float(10.0), *[C.c_int(v) for v in [0, 4, 4, 4]], C.c_void_p(locks.data_ptr())]
                        assert len(values) == 30
                        argv = (C.c_void_p * len(values))(*[C.addressof(v) for v in values])
                        def run():
                            out.zero_()
                            if label.endswith('_private'):
                                assert rows <= cap // 2
                                rc = lib.probe_launch_private(argv, concurrency, rows,
                                    torch.cuda.current_stream().cuda_stream)
                            else:
                                rc = lib.probe_launch(argv, concurrency, torch.cuda.current_stream().cuda_stream)
                            assert rc == 0, f'CUDA launch error {rc}'
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            run()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    samples = []
                    for _ in range(3):
                        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record()
                        for _ in range(args.repeats):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) / args.repeats)
                    assert out.isfinite().all()
                    if baseline is None:
                        baseline = out.clone()
                    delta = out - baseline
                    rmse = (delta.square().mean() / baseline.square().mean()).sqrt().item()
                    assert rmse < 0.005, (label, rmse)
                    if label.startswith('g8') and '_k16' not in label:
                        assert rmse < 1e-6, ('same-math parity failure', label, rmse)
                    result['variants'][label] = {'ms': statistics.median(samples),
                        'estimated_packed_weight_gbps': packed_weight_bytes / statistics.median(samples) / 1e6,
                        'relative_rmse': rmse, 'max_abs_difference': delta.abs().max().item()}
                    del graph
                print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
