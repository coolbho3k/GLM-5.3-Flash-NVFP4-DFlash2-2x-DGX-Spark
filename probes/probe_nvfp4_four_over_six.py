#!/usr/bin/env python3
"""Measure the 288-byte MLA writer's reconstruction error and payload hash.

Run this identical probe in the amax/6 baseline and four-over-six candidate
images.  It launches the real CuteDSL writer, decodes the resulting 288-byte
records on the GPU, and reports per-group SSE plus the stored-scale bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from b12x.attention._shared.mla.kv_cache import (
    concat_and_cache_nvfp4_mla_zero_rope,
)


E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
BLOCK_SIZE = 2304


def make_latents(tokens: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(0x4A6)
    values = torch.randn(
        tokens, 32, 16, generator=generator, device=device, dtype=torch.float32
    )
    # Cover ordinary near-Gaussian groups, varied magnitudes, and the
    # outlier-heavy shapes where mapping amax to 4 can improve the other 15
    # values.  BF16 conversion mirrors the production writer input.
    group_scale = torch.logspace(-3, 1, 32, device=device).view(1, 32, 1)
    values = values * group_scale
    values[1::3, :, 0] *= 4.0
    values[2::3, :, 1:4] *= 2.0
    return values.reshape(tokens, 512).to(torch.bfloat16).contiguous()


def decode_records(
    records: torch.Tensor, latent_scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = records[:, :256].reshape(-1, 32, 8)
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape(-1, 32, 16)
    signs = torch.where((codes & 8) != 0, -1.0, 1.0)
    magnitude = E2M1.to(records.device)[(codes & 7).long()]
    scales = (
        records[:, 256:288]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    ) * latent_scale
    return signs * magnitude * scales.unsqueeze(-1), scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-sizes", default="1,8,384")
    parser.add_argument("--label", default="unknown")
    parser.add_argument("--latent-scale", type=float, default=1.0)
    parser.add_argument("--dump", type=Path)
    args = parser.parse_args()

    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    if args.tokens < 3 or any(size < 1 or size > args.tokens for size in batch_sizes):
        raise ValueError("tokens must be >= 3 and cover every positive batch size")
    if args.iterations < 1 or args.warmup < 1:
        raise ValueError("iterations and warmup must be positive")
    if not math.isfinite(args.latent_scale) or args.latent_scale <= 0.0:
        raise ValueError("latent-scale must be finite and positive")
    device = torch.device("cuda")
    source = make_latents(args.tokens, device)
    cache = torch.full(
        (1, BLOCK_SIZE, 288), 0xA5, device=device, dtype=torch.uint8
    )
    slots = torch.arange(args.tokens, device=device, dtype=torch.int64)
    concat_and_cache_nvfp4_mla_zero_rope(
        source, cache, slots, latent_scale=args.latent_scale
    )
    torch.cuda.synchronize()

    records = cache[0, : args.tokens]
    reconstructed, scales = decode_records(records, args.latent_scale)
    source_groups = source.float().reshape(-1, 32, 16)
    squared = (reconstructed - source_groups).square()
    group_sse = squared.sum(dim=-1)
    payload = records.cpu().numpy().tobytes()
    scale_payload = records[:, 256:288].cpu().numpy().tobytes()
    if args.dump is not None:
        torch.save(
            {
                "group_sse": group_sse.cpu(),
                "scales": records[:, 256:288].cpu(),
                "records": records.cpu(),
            },
            args.dump,
        )

    timings = {}
    for batch_size in batch_sizes:

        def write() -> None:
            concat_and_cache_nvfp4_mla_zero_rope(
                source[:batch_size],
                cache,
                slots[:batch_size],
                latent_scale=args.latent_scale,
            )

        for _ in range(args.warmup):
            write()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            write()
        end.record()
        end.synchronize()
        milliseconds = float(start.elapsed_time(end)) / args.iterations
        timings[str(batch_size)] = {
            "milliseconds": milliseconds,
            "nanoseconds_per_token": milliseconds * 1.0e6 / batch_size,
        }

    print(
        json.dumps(
            {
                "label": args.label,
                "latent_scale": args.latent_scale,
                "global_scale": 1.0 / args.latent_scale,
                "tokens": args.tokens,
                "block_size": BLOCK_SIZE,
                "groups": int(group_sse.numel()),
                "mse": float(squared.mean().item()),
                "mean_group_sse": float(group_sse.mean().item()),
                "max_group_sse": float(group_sse.max().item()),
                "zero_scale_groups": int((scales == 0).sum().item()),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "scale_sha256": hashlib.sha256(scale_payload).hexdigest(),
                "writer_timings": timings,
                "iterations": args.iterations,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
