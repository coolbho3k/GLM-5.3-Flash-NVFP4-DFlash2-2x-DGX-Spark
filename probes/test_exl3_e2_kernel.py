#!/usr/bin/env python3
"""GPU parity checks for MiaAI's E2/E3 EXL3 fat-expert CUDA kernels."""

from __future__ import annotations

import os
import subprocess
import types

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the E2 kernel parity check")

    import exllamav3_ext
    from vllm.model_executor.layers.quantization.exl3 import (
        MCG_MARKER_SIGNED_INT32,
        execute_exl3_linear,
    )

    for symbol in (
        "exl3_moe", "exl3_fat_gemm", "exl3_fat_gemm_scatter",
        "exl3_fat_moe_gather", "exl3_fat_moe_gateup", "exl3_fat_moe_down",
    ):
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
    check_grouped_e3(device)


def check_grouped_e3(device: torch.device) -> None:
    from vllm.model_executor.layers.quantization.exl3 import (
        MCG_MARKER_SIGNED_INT32,
        Exl3Config,
        Exl3MoEMethod,
        apply_exl3_experts,
        exl3_fat_diag,
    )

    keys = (
        "EXL3_FAT_GROUPED", "EXL3_FAT_KERNEL", "EXL3_MOE_ROW_TILE",
        "EXL3_TEMP_ROWS_FUSED", "EXL3_FAT_SCRATCH_ROWS", "EXL3_FAT_EXPERT_LOG",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update({
            "EXL3_FAT_GROUPED": "1",
            "EXL3_FAT_KERNEL": "1",
            "EXL3_MOE_ROW_TILE": "0",
            "EXL3_TEMP_ROWS_FUSED": "32",
            "EXL3_FAT_SCRATCH_ROWS": "256",
            "EXL3_FAT_EXPERT_LOG": "0",
        })
        method = Exl3MoEMethod(types.SimpleNamespace(swiglu_limit=10.0), Exl3Config())
        layer = torch.nn.Module()
        method.create_weights(
            layer, num_experts=3, hidden_size=256,
            intermediate_size_per_partition=256, params_dtype=torch.float16,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(73)
        with torch.no_grad():
            layer.w13_trellis.copy_(torch.randint(
                -30000, 30000, tuple(layer.w13_trellis.shape),
                dtype=torch.int16, generator=generator,
            ))
            layer.w2_trellis.copy_(torch.randint(
                -30000, 30000, tuple(layer.w2_trellis.shape),
                dtype=torch.int16, generator=generator,
            ))
            layer.w13_suh.copy_(torch.randn(
                tuple(layer.w13_suh.shape), generator=generator,
            ).half())
            layer.w13_suh[:, 1].copy_(layer.w13_suh[:, 0])
            layer.w13_svh.copy_(torch.randn(
                tuple(layer.w13_svh.shape), generator=generator,
            ).half())
            layer.w2_suh.copy_(torch.randn(
                tuple(layer.w2_suh.shape), generator=generator,
            ).half())
            layer.w2_svh.copy_(torch.randn(
                tuple(layer.w2_svh.shape), generator=generator,
            ).half())
            layer.w13_mcg.fill_(MCG_MARKER_SIGNED_INT32)
            layer.w2_mcg.fill_(MCG_MARKER_SIGNED_INT32)
        layer = layer.to(device)
        method.process_weights_after_loading(layer)
        assert layer._exl3_fat_effective_tier == "grouped", (
            layer._exl3_fat_effective_tier, layer._exl3_fat_tier_reason,
        )

        rows = 160
        x = torch.randn(rows, 256, dtype=torch.float16, device=device)
        ids = torch.zeros(rows, 2, dtype=torch.long, device=device)
        ids[:, 1] = 1
        weights = torch.full((rows, 2), 0.5, dtype=torch.float16, device=device)
        reference = apply_exl3_experts(x, ids, weights, layer, fused=False)
        grouped = apply_exl3_experts(x, ids, weights, layer, fused=True)
        error = float((reference.float() - grouped.float()).abs().max())
        bound = max(0.15, 0.08 * float(reference.float().abs().max().clamp_min(1.0)))
        assert torch.isfinite(grouped).all()
        assert error < bound, (error, bound)
        assert layer._exl3_last_fat_fallback == "grouped"
        diag = exl3_fat_diag()
        assert diag["sym_fat_moe"] and diag["grouped_calls"] >= 1
        print(f"EXL3 E3 grouped parity OK rows={rows} error={error:.5f} bound={bound:.5f}")
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
