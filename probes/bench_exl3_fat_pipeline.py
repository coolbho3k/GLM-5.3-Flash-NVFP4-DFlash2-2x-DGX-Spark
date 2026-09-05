#!/usr/bin/env python3
"""Isolated E2 pipeline parity/timing. Stop the serving cluster before running.

Tests actual per-rank gate/up and down dimensions, M tails, short/odd K loops,
direct output and unique-index scatter into existing output. No model-quality
claim: weights and inputs are synthetic, compared with the installed kernel.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics

import torch
import exllamav3_ext as ext


def timed(run, repeats):
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
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    del graph
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libraries', type=Path, required=True)
    parser.add_argument('--variants', default='control,async2')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--rows', default='129,145,512,2048,8192',
                        help='Real projection row counts; boundary cases always run first')
    parser.add_argument('--small-only', action='store_true',
                        help='Only boundary cases, suitable for sanitizer checks')
    parser.add_argument('--native-pipelines', action='store_true',
                        help='Also validate/time the installed additive native symbols')
    args = parser.parse_args()
    assert args.repeats > 0
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.04)
    torch.manual_seed(73)
    libraries = {}
    for name in args.variants.split(','):
        lib = C.CDLL(str((args.libraries / (name + '.so')).resolve()))
        lib.fat_probe_launch.restype = C.c_int
        lib.fat_probe_launch.argtypes = [C.POINTER(C.c_void_p), *([C.c_int]*4), C.c_void_p]
        lib.fat_probe_occupancy.argtypes = [C.c_int]
        libraries[name] = lib
    installed = {}
    if args.native_pipelines:
        assert ext.glm53_fat_pipeline_version() == 2
        for mode, suffix in [('m128', '_async2'), ('m64', '_async2_m64')]:
            installed['extension_' + mode] = (getattr(ext, 'exl3_fat_gemm' + suffix),
                                              getattr(ext, 'exl3_fat_gemm_scatter' + suffix))
    shapes = [(1, 16, 128, False), (127, 48, 256, False),
              (128, 32, 128, True), (129, 48, 256, True)]
    for m in ([] if args.small_only else map(int, args.rows.split(','))):
        assert 0 < m <= 8192
        shapes += [(m, 4096, 2048, False), (m, 1024, 4096, True)]
    with torch.inference_mode():
        for m, k, n, scatter in shapes:
            a = torch.randn(m, k, dtype=torch.float16, device='cuda') * 0.2
            packed = torch.randint(-32768, 32767, (k//16, n//16, 64),
                                   dtype=torch.int16, device='cuda')
            svh = torch.randn(n, dtype=torch.float16, device='cuda') * 0.03
            idx = torch.randperm(m + 17, device='cuda')[:m].contiguous()
            weights = torch.rand(m, dtype=torch.float16, device='cuda')
            out = torch.empty(m + 17 if scatter else m, n, dtype=torch.float32, device='cuda')

            def native():
                if scatter:
                    ext.exl3_fat_gemm_scatter(a, packed, out, svh, idx, weights, 4, True, False)
                else:
                    ext.exl3_fat_gemm(a, packed, out, svh, 4, True, False)

            # Nonzero output checks scatter's += contract and untouched rows.
            out.fill_(0.125 if scatter else float('nan'))
            native()
            reference = out.clone()
            row = dict(m=m, k=k, n=n, scatter=scatter, repeats=args.repeats, seed=73, variants={})
            for name in ['native', *libraries, *installed, 'native_repeat']:
                holders = argv = None
                if name.startswith('native'):
                    call = native
                    metadata = {}
                elif name in installed:
                    direct_fn, scatter_fn = installed[name]
                    metadata = {'installed_extension': True}

                    def call():
                        if scatter:
                            scatter_fn(a, packed, out, svh, idx, weights, 4, True, False)
                        else:
                            direct_fn(a, packed, out, svh, 4, True, False)
                else:
                    lib = libraries[name]
                    metadata = dict(shared_bytes=lib.fat_probe_shared_bytes(),
                                    resident_blocks_per_sm=lib.fat_probe_occupancy(int(scatter)))
                    if hasattr(lib, 'fat_probe_tile_m'):
                        metadata['tile_m'] = lib.fat_probe_tile_m()
                    assert metadata['resident_blocks_per_sm'] > 0
                    holders = [C.c_void_p(t.data_ptr()) for t in [a, packed, out, svh]]
                    holders += [C.c_void_p(idx.data_ptr() if scatter else 0),
                                C.c_void_p(weights.data_ptr() if scatter else 0)]
                    holders += [C.c_int(v) for v in [m, k, n]]
                    argv = (C.c_void_p * len(holders))(*[C.addressof(v) for v in holders])

                    def call():
                        rc = lib.fat_probe_launch(argv, m, k, n, int(scatter),
                                                  torch.cuda.current_stream().cuda_stream)
                        assert rc == 0, (name, rc)

                out.fill_(0.125 if scatter else float('nan'))
                call()
                torch.cuda.synchronize()
                assert out.isfinite().all(), (name, 'nonfinite output')
                delta = out - reference
                rel = (delta.square().mean() / reference.square().mean().clamp_min(1e-30)).sqrt().item()
                assert rel < 1e-6, (name, m, k, n, rel)
                bitwise = torch.equal(out, reference)
                maximum = delta.abs().max().item()

                def run():
                    if scatter:
                        out.fill_(0.125)
                    call()

                ms = timed(run, args.repeats)
                assert out.isfinite().all()
                replay_rel = ((out - reference).square().mean() /
                              reference.square().mean().clamp_min(1e-30)).sqrt().item()
                assert replay_rel < 1e-6, ('replay parity', name, replay_rel)
                row['variants'][name] = dict(ms=ms, relative_rmse=rel,
                    replay_relative_rmse=replay_rel, bitwise_equal=bitwise,
                    max_abs_difference=maximum, **metadata)
            print(json.dumps(row), flush=True)
            del a, packed, svh, idx, weights, out, reference, delta


if __name__ == '__main__':
    main()
