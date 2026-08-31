#!/usr/bin/env python3
"""Compare the compatible and GLM-native NVFP4 sparse-MLA layouts on GPU.

Stage 5A stores the 512-value NVFP4 latent in a 368-byte compatibility record
whose 80-byte FP8-RoPE region is retained. Stage 5B removes that region for
GLM's qk_rope_head_dim=0 geometry, yielding a strict 288-byte record. The probe
uses a zero RoPE query, so both layouts must contain identical latent bytes and
produce equivalent native SM121 sparse-attention results even though Stage 5A
also carries a nonzero compatibility payload.
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from b12x.attention._shared.mla.kv_cache import (
    concat_and_cache_nvfp4_mla_fp8_rope,
    concat_and_cache_nvfp4_mla_zero_rope,
)
from b12x.attention._shared.mla.prefill import run_unified_prefill
from b12x.attention._shared.mla.traits import ScaleFormat


def elapsed_ms(operation, iterations: int) -> float:
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--topk", type=int, default=2176)
    args = parser.parse_args()

    torch.manual_seed(31)
    device = torch.device("cuda")
    block_size = 64
    topk = args.topk
    blocks = (topk + block_size - 1) // block_size
    slots = blocks * block_size
    query_tokens = 2
    # GLM TP2 presents 32 local query heads to the native sparse kernel. Using
    # that production partition also avoids conflating layout validation with
    # the separate padded eight-head tail specialization.
    heads = 32

    latent = (
        torch.randn(slots, 512, device=device, dtype=torch.bfloat16) * 0.2
    ).contiguous()
    compatibility_rope = (
        torch.randn(slots, 64, device=device, dtype=torch.bfloat16) * 0.2
    ).contiguous()
    slot_mapping = torch.arange(slots, device=device, dtype=torch.int64)
    compatible_cache = torch.empty(
        blocks, block_size, 368, device=device, dtype=torch.uint8
    )
    native_cache = torch.empty(
        blocks, block_size, 288, device=device, dtype=torch.uint8
    )

    def write_compatible() -> None:
        concat_and_cache_nvfp4_mla_fp8_rope(
            latent, compatibility_rope, compatible_cache, slot_mapping
        )

    def write_native() -> None:
        concat_and_cache_nvfp4_mla_zero_rope(
            latent, native_cache, slot_mapping
        )

    write_compatible()
    write_native()
    torch.cuda.synchronize()

    latent_mismatches = int(
        torch.count_nonzero(compatible_cache[..., :288] != native_cache).item()
    )
    compatible_rope_nonzero = int(
        torch.count_nonzero(compatible_cache[..., 288:]).item()
    )
    if latent_mismatches:
        raise AssertionError(
            f"compatible/native latent payload differs in {latent_mismatches} bytes"
        )
    if compatible_rope_nonzero == 0:
        raise AssertionError("compatibility RoPE payload was not written")

    q_native = (
        torch.randn(
            query_tokens, heads, 512, device=device, dtype=torch.bfloat16
        )
        * 0.2
    ).contiguous()
    q_compatible = torch.cat(
        (
            q_native,
            torch.zeros(
                query_tokens,
                heads,
                64,
                device=device,
                dtype=torch.bfloat16,
            ),
        ),
        dim=-1,
    ).contiguous()
    indices = torch.stack(
        (
            torch.arange(0, topk, device=device, dtype=torch.int32),
            torch.arange(slots - topk, slots, device=device, dtype=torch.int32),
        )
    )

    def read_compatible() -> tuple[torch.Tensor, torch.Tensor]:
        return run_unified_prefill(
            q=q_compatible,
            kv_cache=compatible_cache,
            topk_indices=indices,
            sm_scale=1.0 / math.sqrt(512),
            page_block_size=block_size,
            scale_format=ScaleFormat.NVFP4_E4M3,
            fp8_rope=True,
        )

    def read_native() -> tuple[torch.Tensor, torch.Tensor]:
        return run_unified_prefill(
            q=q_native,
            kv_cache=native_cache,
            topk_indices=indices,
            sm_scale=1.0 / math.sqrt(512),
            page_block_size=block_size,
            scale_format=ScaleFormat.NVFP4_E4M3,
            fp8_rope=False,
        )

    compatible_output, compatible_lse = read_compatible()
    native_output, native_lse = read_native()
    torch.cuda.synchronize()
    finite_counts = {
        "compatible_output": int(torch.isfinite(compatible_output).sum().item()),
        "compatible_output_total": compatible_output.numel(),
        "native_output": int(torch.isfinite(native_output).sum().item()),
        "native_output_total": native_output.numel(),
        "compatible_lse": int(torch.isfinite(compatible_lse).sum().item()),
        "compatible_lse_total": compatible_lse.numel(),
        "native_lse": int(torch.isfinite(native_lse).sum().item()),
        "native_lse_total": native_lse.numel(),
    }
    if any(
        finite_counts[name] != finite_counts[name + "_total"]
        for name in (
            "compatible_output",
            "native_output",
            "compatible_lse",
            "native_lse",
        )
    ):
        raise AssertionError(f"non-finite native result: {finite_counts}")

    output_error = (compatible_output.float() - native_output.float()).abs()
    lse_error = (compatible_lse.float() - native_lse.float()).abs()
    output_max_abs = float(output_error.max().item())
    output_mean_abs = float(output_error.mean().item())
    lse_max_abs = float(lse_error.max().item())
    # Both kernels consume bit-identical quantized latents. A small BF16
    # reduction-order difference is permitted between their distinct traits.
    if output_max_abs > 0.0625 or output_mean_abs > 0.002:
        raise AssertionError(
            f"layout output mismatch: max={output_max_abs}, mean={output_mean_abs}"
        )
    if lse_max_abs > 0.01:
        raise AssertionError(f"layout LSE mismatch: max={lse_max_abs}")

    compatible_write_ms = elapsed_ms(write_compatible, args.iterations)
    native_write_ms = elapsed_ms(write_native, args.iterations)
    compatible_read_ms = elapsed_ms(read_compatible, args.iterations)
    native_read_ms = elapsed_ms(read_native, args.iterations)

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "compatible_record_bytes": 368,
                "native_record_bytes": 288,
                "storage_reduction": 1.0 - 288.0 / 368.0,
                "latent_mismatches": latent_mismatches,
                "compatible_rope_nonzero": compatible_rope_nonzero,
                "output_max_abs": output_max_abs,
                "output_mean_abs": output_mean_abs,
                "lse_max_abs": lse_max_abs,
                "compatible_write_ms": compatible_write_ms,
                "native_write_ms": native_write_ms,
                "compatible_read_ms": compatible_read_ms,
                "native_read_ms": native_read_ms,
                "iterations": args.iterations,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
