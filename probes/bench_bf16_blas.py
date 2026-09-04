#!/usr/bin/env python3
"""Small, bounded BF16 linear backend screen; run inside the serving image.

Run only with the serving processes stopped on the test GPU. Synthetic weights
match representative TP2 shapes, not model activations. Kernel timings alone
are not a serving-speed claim. All backends see identical BF16 inputs.
"""
import argparse
import json
import statistics

import torch
import torch.nn.functional as F


def measure(x, weight, repeats):
    # Capturing amortizes Python launches, as in the C1 target graph.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            F.linear(x, weight)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = F.linear(x, weight)
    samples = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    return out.clone(), statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', default='8,48,280,2048')
    parser.add_argument('--shapes', default='12576x4096,4096x4096,12288x4096,4096x6144,4096x1024')
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--column-major', action='store_true',
                        help='Also screen a lossless change in weight storage layout')
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    print(json.dumps({'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
                      'bf16_reduced_precision_reduction':
                      torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                      'args': vars(args)}), flush=True)
    with torch.inference_mode():
        for shape in args.shapes.split(','):
            n, k = map(int, shape.split('x'))
            weight = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
            column_weight = weight.T.contiguous().T if args.column_major else None
            for m in map(int, args.rows.split(',')):
                x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
                # A small FP32 reference slice avoids a large temporary allocation.
                torch.backends.cuda.matmul.allow_tf32 = False
                reference = F.linear(x[:8].float(), weight[:256].float())
                results = {}
                baseline = None
                # Repeat the control after Lt to expose clock/cache drift.
                candidates = [('cublas', 'cublas', weight), ('cublaslt', 'cublaslt', weight)]
                if column_weight is not None:
                    candidates += [('column_cublas', 'cublas', column_weight),
                                   ('column_cublaslt', 'cublaslt', column_weight)]
                candidates += [('cublas_repeat', 'cublas', weight)]
                for label, backend, candidate_weight in candidates:
                    torch.backends.cuda.preferred_blas_library(backend)
                    out, latency = measure(x, candidate_weight, args.repeats)
                    error = out[:8, :256].float() - reference
                    relative_rmse = (error.square().mean() / reference.square().mean()).sqrt().item()
                    if baseline is None:
                        baseline = out.clone()
                    results[label] = {'us': latency, 'relative_rmse_to_fp32_slice': relative_rmse,
                                      'max_difference_to_control': (out - baseline).abs().max().item(),
                                      'finite': bool(out.isfinite().all().item())}
                    del out
                print(json.dumps({'m': m, 'n': n, 'k': k, 'results': results}), flush=True)
                del baseline, x, reference
            del weight
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
