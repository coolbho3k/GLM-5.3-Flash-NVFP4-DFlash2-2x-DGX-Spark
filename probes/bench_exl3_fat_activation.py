#!/usr/bin/env python3
"""Check E2's existing activation against vLLM's fused FP32 clamp/SwiGLU."""
import json
import argparse
import torch
import vllm._custom_ops  # Register the packaged CUDA operators.


def timed(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(100):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 10  # microseconds per replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integration', action='store_true',
                        help='Test the exact installed EXL3 helper, not just the primitive')
    args = parser.parse_args()
    if args.integration:
        from vllm.model_executor.layers.quantization.exl3 import exl3_fat_activation
    torch.manual_seed(42)
    torch.set_num_threads(2)
    with torch.inference_mode():
        for rows in (145, 512, 2048, 8192):
            width, limit = 1024, 10.0
            source = torch.randn(rows, 2 * width, dtype=torch.float32, device='cuda') * 20
            # Include clamp boundaries, tiny gates, and both activation tails.
            source[0, :8] = torch.tensor([-100, -10, -1e-7, 0, 1e-7, 9.999, 10, 100], device='cuda')
            x = source.clone()
            act = torch.empty(rows, width, device='cuda', dtype=torch.float32)
            half = torch.empty_like(act, dtype=torch.float16)
            fused = torch.empty_like(act)
            fused_half = torch.empty_like(half)

            def baseline():
                if args.integration:
                    return exl3_fat_activation(x, act, half, limit, fused=False)
                gate, up = x[:, :width], x[:, width:]
                gate.clamp_(max=limit)
                up.clamp_(min=-limit, max=limit)
                torch.sigmoid(gate, out=act)
                act.mul_(gate).mul_(up)
                half.copy_(act)

            def candidate():
                if args.integration:
                    return exl3_fat_activation(source, fused, fused_half, limit, fused=True)
                torch.ops._C.silu_and_mul_with_clamp(fused, source, limit, 1.0, 0.0)
                fused_half.copy_(fused)

            baseline()
            candidate()
            difference = (fused - act).abs()
            assert fused.isfinite().all()
            torch.testing.assert_close(fused, act, atol=3e-5, rtol=3e-6)
            # FP16 rounding can differ by one ULP around a rounding boundary.
            torch.testing.assert_close(fused_half, half, atol=6e-5, rtol=1e-3)
            result = {'rows': rows, 'width': width,
                      'max_fp32_difference': difference.max().item(),
                      'half_mismatch_fraction': (half != fused_half).float().mean().item(),
                      'max_half_difference': (half - fused_half).abs().max().item(),
                      'baseline_us': timed(baseline), 'fused_us': timed(candidate)}
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
