#!/usr/bin/env python3
"""GPU numerical test for the 528-byte GLM NoPE FP8 cache writer."""

from __future__ import annotations

import torch

from b12x.attention._shared.mla.fp8_zero_rope import (
    concat_and_cache_fp8_mla_zero_rope,
)


def reference(row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = row.float().reshape(4, 128)
    scales = grouped.abs().amax(dim=1) / 448.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quant = (grouped / scales[:, None]).clamp(-448.0, 448.0)
    return quant.to(torch.float8_e4m3fn).float().reshape(512), scales


def main() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(53)
    source = (torch.randn(5, 512, device="cuda") * 1.75).to(torch.bfloat16)
    source[1].zero_()
    cache = torch.zeros((3, 4, 528), dtype=torch.uint8, device="cuda")
    slots = torch.tensor([0, 3, 4, -1, 11], dtype=torch.int64, device="cuda")
    concat_and_cache_fp8_mla_zero_rope(source, cache, slots)
    torch.cuda.synchronize()

    for source_row, slot in ((0, 0), (1, 3), (2, 4), (4, 11)):
        block, offset = divmod(slot, 4)
        record = cache[block, offset]
        actual_q = record[:512].view(torch.float8_e4m3fn).float()
        actual_scales = record[512:].view(torch.float32)
        expected_q, expected_scales = reference(source[source_row])
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(actual_scales, expected_scales, rtol=1e-6, atol=0)

    # The skipped row must not alias slot zero or modify any unwritten record.
    written = {0, 3, 4, 11}
    flat = cache.reshape(12, 528)
    for slot in set(range(12)) - written:
        assert not flat[slot].any(), f"unexpected write to slot {slot}"
    print("528-byte zero-RoPE FP8 writer matches the PyTorch reference")


if __name__ == "__main__":
    main()
