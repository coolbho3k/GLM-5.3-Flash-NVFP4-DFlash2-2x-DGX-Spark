#!/usr/bin/env python3
"""Exercise DFlash2's B12X MXFP8 linear shapes under CUDA graph replay.

The full GLM target is deliberately not needed.  Each unique draft weight
geometry is loaded and tested independently so a graph-safety failure can be
localized with little unified-memory pressure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

from b12x.gemm import mxfp8_linear


WEIGHTS = (
    "candidate_selector.hidden_projection.weight",
    "layers.0.attention_conv.kernel_projection.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "fc.weight",
)


def load_pair(checkpoint: Path, name: str, device: torch.device):
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(name).to(device)
        scale = handle.get_tensor(f"{name}_scale").to(device)
    return weight, scale


def run_one(checkpoint: Path, name: str, m: int, device: torch.device) -> None:
    weight, scale = load_pair(checkpoint, name, device)
    packed = mxfp8_linear.pack_weight(weight, scale)
    k = int(weight.shape[1])
    source = torch.randn((m, k), dtype=torch.bfloat16, device=device)

    # PyTorch requires warmup on a side stream before capture.  This also
    # resolves every B12X/CuTe JIT specialization outside the graph.
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            mxfp8_linear.mm(
                source,
                packed,
                expected_m=m,
                stream=warmup_stream.cuda_stream,
            )
    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mxfp8_linear.mm(
            source,
            packed,
            expected_m=m,
            stream=torch.cuda.current_stream(device).cuda_stream,
        )
    torch.cuda.synchronize(device)

    # Change the static input in place, replay twice, and compare with eager.
    source.normal_()
    graph.replay()
    torch.cuda.synchronize(device)
    replay = captured.clone()
    eager = mxfp8_linear.mm(
        source,
        packed,
        expected_m=m,
        stream=torch.cuda.current_stream(device).cuda_stream,
    )
    torch.cuda.synchronize(device)

    diff = replay.float() - eager.float()
    max_abs = float(diff.abs().max())
    rmse = float(diff.square().mean().sqrt())
    if not torch.isfinite(replay).all():
        raise AssertionError(f"{name}: graph replay produced non-finite output")
    if not torch.equal(replay, eager):
        raise AssertionError(
            f"{name}: graph/eager mismatch max_abs={max_abs:.6g} rmse={rmse:.6g}"
        )
    print(
        f"PASS {name} M={m} N={weight.shape[0]} K={weight.shape[1]} "
        f"max_abs={max_abs:.1f} rmse={rmse:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--weight", action="append", dest="weights")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.m < 1:
        raise ValueError("--m must be positive")
    checkpoint = args.checkpoint / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = torch.device("cuda:0")
    for name in args.weights or WEIGHTS:
        run_one(checkpoint, name, args.m, device)


if __name__ == "__main__":
    main()
