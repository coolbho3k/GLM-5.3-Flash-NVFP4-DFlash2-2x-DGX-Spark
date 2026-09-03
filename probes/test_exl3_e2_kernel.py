#!/usr/bin/env python3
"""GPU parity check for MiaAI's E2 EXL3 fat-expert CUDA kernel."""

from __future__ import annotations

import subprocess

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the E2 kernel parity check")

    import exllamav3_ext
    from vllm.model_executor.layers.quantization.exl3 import (
        MCG_MARKER_SIGNED_INT32,
        execute_exl3_linear,
    )

    for symbol in ("exl3_moe", "exl3_fat_gemm", "exl3_fat_gemm_scatter"):
        assert hasattr(exllamav3_ext, symbol), (symbol, dir(exllamav3_ext))
    cubins = subprocess.check_output(
        ["cuobjdump", "-lelf", exllamav3_ext.__file__],
        text=True,
        stderr=subprocess.STDOUT,
    ).lower()
    assert "sm_121" in cubins or "compute_121" in cubins, cubins[-2000:]

    device = torch.device("cuda:0")
    rows = 145
    width = 256
    generator = torch.Generator(device="cpu")
    generator.manual_seed(41)
    trellis = torch.randint(
        -30000,
        30000,
        (width // 16, width // 16, 4 * 16),
        dtype=torch.int16,
        generator=generator,
    ).to(device)
    suh = torch.where(
        torch.rand(width, generator=generator) > 0.5,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ).half().to(device)
    svh = torch.where(
        torch.rand(width, generator=generator) > 0.5,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ).half().to(device)
    marker = torch.tensor(
        [MCG_MARKER_SIGNED_INT32], dtype=torch.int32, device=device
    )
    inputs = torch.randn(rows, width, dtype=torch.float16, device=device)
    reference = execute_exl3_linear(
        inputs, trellis, suh, svh, marker, out_dtype=torch.float32
    )

    transformed = torch.empty_like(inputs)
    exllamav3_ext.had_r_128(inputs, transformed, suh, None, 1.0)
    direct = torch.empty(rows, width, dtype=torch.float32, device=device)
    exllamav3_ext.exl3_fat_gemm(
        transformed, trellis, direct, svh, 4, True, False
    )
    bound = max(
        0.15, 0.08 * float(reference.float().abs().max().clamp_min(1.0))
    )
    direct_error = float((reference - direct).abs().max())
    assert torch.isfinite(direct).all()
    assert direct_error < bound, (direct_error, bound)

    token_idx = torch.randperm(rows + 17, device=device)[:rows].contiguous()
    route_weight = torch.rand(rows, dtype=torch.float16, device=device)
    expected = torch.zeros(rows + 17, width, dtype=torch.float32, device=device)
    expected.index_add_(
        0, token_idx, reference * route_weight.float().unsqueeze(-1)
    )
    scattered = torch.zeros_like(expected)
    exllamav3_ext.exl3_fat_gemm_scatter(
        transformed,
        trellis,
        scattered,
        svh,
        token_idx,
        route_weight,
        4,
        True,
        False,
    )
    scatter_error = float((expected - scattered).abs().max())
    assert torch.isfinite(scattered).all()
    assert scatter_error < bound, (scatter_error, bound)
    print(
        "EXL3 E2 direct/scatter parity OK "
        f"rows={rows} direct={direct_error:.5f} "
        f"scatter={scatter_error:.5f} bound={bound:.5f}"
    )


if __name__ == "__main__":
    main()
