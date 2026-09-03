"""Native 528-byte GLM NoPE FP8 MLA cache writer for SM121.

Each token record stores 512 E4M3 latent bytes followed by four FP32
dequantization scales. GLM-5.3 has qk_rope_head_dim == 0, so the generic
128-byte zero-filled RoPE payload is deliberately omitted.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_HEAD_SIZE = 512
_NUM_GROUPS = 4
_RECORD_BYTES = 528
_FP8_MAX = 448.0


@triton.jit
def _concat_and_cache_fp8_mla_zero_rope_kernel(
    kv_c,
    cache_fp8,
    cache_f32,
    slot_mapping,
    num_tokens,
    kv_stride_t: tl.constexpr,
    cache_fp8_stride_block: tl.constexpr,
    cache_fp8_stride_token: tl.constexpr,
    cache_f32_stride_block: tl.constexpr,
    cache_f32_stride_token: tl.constexpr,
    block_size: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slot_mapping + token)
    valid = (token < num_tokens) & (slot >= 0)

    dims = tl.arange(0, 512)
    values = tl.load(
        kv_c + token * kv_stride_t + dims,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    magnitude = tl.abs(values)

    amax0 = tl.max(tl.where(dims < 128, magnitude, 0.0), axis=0)
    amax1 = tl.max(
        tl.where((dims >= 128) & (dims < 256), magnitude, 0.0), axis=0
    )
    amax2 = tl.max(
        tl.where((dims >= 256) & (dims < 384), magnitude, 0.0), axis=0
    )
    amax3 = tl.max(tl.where(dims >= 384, magnitude, 0.0), axis=0)

    scale0 = tl.where(amax0 > 0.0, amax0 / 448.0, 1.0)
    scale1 = tl.where(amax1 > 0.0, amax1 / 448.0, 1.0)
    scale2 = tl.where(amax2 > 0.0, amax2 / 448.0, 1.0)
    scale3 = tl.where(amax3 > 0.0, amax3 / 448.0, 1.0)
    scale = tl.where(
        dims < 128,
        scale0,
        tl.where(dims < 256, scale1, tl.where(dims < 384, scale2, scale3)),
    )
    quant = tl.maximum(tl.minimum(values / scale, 448.0), -448.0).to(
        tl.float8e4nv
    )

    block = slot // block_size
    offset = slot - block * block_size
    fp8_base = block * cache_fp8_stride_block + offset * cache_fp8_stride_token
    tl.store(cache_fp8 + fp8_base + dims, quant, mask=valid)

    f32_base = block * cache_f32_stride_block + offset * cache_f32_stride_token
    groups = tl.arange(0, 4)
    scales = tl.where(
        groups == 0,
        scale0,
        tl.where(groups == 1, scale1, tl.where(groups == 2, scale2, scale3)),
    )
    tl.store(cache_f32 + f32_base + 128 + groups, scales, mask=valid)


def _launch(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens = int(slot_mapping.numel())
    if num_tokens == 0:
        return
    cache_fp8 = kv_cache.view(torch.float8_e4m3fn)
    cache_f32 = kv_cache.view(torch.float32)
    _concat_and_cache_fp8_mla_zero_rope_kernel[(num_tokens,)](
        kv_c,
        cache_fp8,
        cache_f32,
        slot_mapping,
        num_tokens=num_tokens,
        kv_stride_t=kv_c.stride(0),
        cache_fp8_stride_block=cache_fp8.stride(0),
        cache_fp8_stride_token=cache_fp8.stride(1),
        cache_f32_stride_block=cache_f32.stride(0),
        cache_f32_stride_token=cache_f32.stride(1),
        block_size=kv_cache.shape[1],
        num_warps=8,
    )


@torch.library.custom_op(
    "b12x::concat_and_cache_fp8_mla_zero_rope",
    mutates_args=("kv_cache",),
)
def _concat_and_cache_fp8_mla_zero_rope_op(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    _launch(kv_c, kv_cache, slot_mapping)


@_concat_and_cache_fp8_mla_zero_rope_op.register_fake
def _concat_and_cache_fp8_mla_zero_rope_fake(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    return None


def concat_and_cache_fp8_mla_zero_rope(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Quantize and write GLM NoPE MLA rows into a 528-byte paged cache."""

    if kv_c.ndim != 2 or kv_c.shape[1] != _HEAD_SIZE:
        raise ValueError(f"kv_c must have shape [tokens, 512], got {tuple(kv_c.shape)}")
    if kv_c.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"kv_c must be bf16/fp16, got {kv_c.dtype}")
    if kv_cache.dtype != torch.uint8 or kv_cache.ndim != 3:
        raise ValueError("kv_cache must be uint8 [blocks, block_size, 528]")
    if kv_cache.shape[2] != _RECORD_BYTES:
        raise ValueError(
            f"zero-RoPE FP8 cache record must be {_RECORD_BYTES} bytes, "
            f"got {kv_cache.shape[2]}"
        )
    if not kv_cache.is_contiguous():
        raise ValueError("zero-RoPE FP8 cache must be contiguous")
    slots = slot_mapping.flatten()
    if slots.dtype != torch.int64:
        raise TypeError(f"slot_mapping must be int64, got {slots.dtype}")
    if slots.numel() > kv_c.shape[0]:
        raise ValueError("slot_mapping has more entries than kv_c rows")
    torch.ops.b12x.concat_and_cache_fp8_mla_zero_rope(
        kv_c[: slots.numel()], kv_cache, slots
    )


__all__ = ["concat_and_cache_fp8_mla_zero_rope"]
