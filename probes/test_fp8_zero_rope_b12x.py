#!/usr/bin/env python3
"""GPU integration test for the 528-byte writer plus B12X MLA reader."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from b12x.attention._shared.mla.fp8_zero_rope import (
    concat_and_cache_fp8_mla_zero_rope,
)
from b12x.attention._shared.mla.prefill import run_unified_prefill
from b12x.attention._shared.mla.traits import ScaleFormat


def main() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(5302)
    device = torch.device("cuda")
    page_size = 64
    # Match the live vLLM warm-up geometry: GLM's requested top-k 2048 is
    # padded to 2176, and TP2 leaves 64 local query heads.
    topk = 2176
    heads = 64

    latent = (torch.randn(topk, 512, device=device) * 0.5).to(torch.bfloat16)
    query = (torch.randn(1, heads, 512, device=device) * 0.5).to(torch.bfloat16)
    cache = torch.zeros(
        (topk // page_size, page_size, 528),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(topk, dtype=torch.int64, device=device)
    concat_and_cache_fp8_mla_zero_rope(latent, cache, slots)

    indices = torch.arange(topk, dtype=torch.int32, device=device).view(1, topk)
    lengths = torch.full((1,), topk, dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(512)
    output, lse = run_unified_prefill(
        q=query,
        kv_cache=cache,
        topk_indices=indices,
        topk_length=lengths,
        sm_scale=sm_scale,
        page_block_size=page_size,
        scale_format=ScaleFormat.ARBITRARY_FP32,
        fp8_rope=False,
    )
    torch.cuda.synchronize()

    qbytes = cache[..., :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = cache[..., 512:].contiguous().view(torch.float32)
    decoded = (
        qbytes.reshape(-1, 4, 128) * scales.reshape(-1, 4, 1)
    ).reshape(-1, 512)
    scores = torch.einsum("thd,kd->thk", query.float(), decoded) * sm_scale
    expected = torch.einsum(
        "thk,kd->thd", torch.softmax(scores, dim=-1), decoded
    )

    cosine = F.cosine_similarity(
        output.float().flatten(), expected.flatten(), dim=0
    ).item()
    max_abs = (output.float() - expected).abs().max().item()
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()
    assert cosine > 0.995, f"B12X zero-RoPE output cosine={cosine:.8f}"
    assert max_abs < 0.03, f"B12X zero-RoPE output max_abs={max_abs:.8f}"
    print(
        "528-byte zero-RoPE B12X reader matches reference: "
        f"cosine={cosine:.8f} max_abs={max_abs:.8f}"
    )


if __name__ == "__main__":
    main()
