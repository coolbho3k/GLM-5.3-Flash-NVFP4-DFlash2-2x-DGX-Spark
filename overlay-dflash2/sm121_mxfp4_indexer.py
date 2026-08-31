"""Fused MXFP4 GLM indexer kernels for GB10 / SM121.

The upstream vLLM FP4 indexer path is restricted to SM100 datacenter GPUs.
GLM-5.3 uses the same compact cache payload (64 packed E2M1 bytes plus four
UE8M0 scale bytes for a 128-value key), but its kpool writer and paged reader
also need to run on SM121.  The physical page follows vLLM's indexer gather
contract: all value payloads first, followed by all per-token scale payloads.
These kernels keep unpacking inside the scoring kernel; no BF16/FP16 copy of
the cache is materialized.

This module intentionally owns only the numerical leaves.  vLLM continues to
own cache allocation, block tables, prefix/rollback metadata, kpool tails, and
top-k selection.
"""

from __future__ import annotations

from threading import Lock

import torch
import triton
import triton.language as tl


HEAD_DIM = 128
MXFP4_BLOCK_SIZE = 32
PACKED_BYTES = HEAD_DIM // 2
SCALE_BYTES = HEAD_DIM // MXFP4_BLOCK_SIZE
RECORD_BYTES = PACKED_BYTES + SCALE_BYTES

_OUTPUTS: dict[tuple[str, int, int, int], torch.Tensor] = {}
_OUTPUTS_LOCK = Lock()


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    """Pack two FP32 values as low/high E2M1 nibbles."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _e2m1_to_fp32(nibble):
    """Decode an unsigned four-bit E2M1 payload to FP32."""
    code = nibble & 0x7
    magnitude = tl.where(
        code == 0,
        0.0,
        tl.where(
            code == 1,
            0.5,
            tl.where(
                code == 2,
                1.0,
                tl.where(
                    code == 3,
                    1.5,
                    tl.where(
                        code == 4,
                        2.0,
                        tl.where(code == 5, 3.0, tl.where(code == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where((nibble & 0x8) != 0, -magnitude, magnitude)


@triton.jit
def _unpack_e2m1(packed, parity):
    nibble = (packed >> (parity * 4)) & 0xF
    return _e2m1_to_fp32(nibble)


@triton.jit
def _fwht_stage(x, groups: tl.constexpr, stride: tl.constexpr):
    x3 = tl.reshape(x, (groups, 2, stride))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (128,))


@triton.jit
def _hadamard128(x):
    x = _fwht_stage(x, 64, 1)
    x = _fwht_stage(x, 32, 2)
    x = _fwht_stage(x, 16, 4)
    x = _fwht_stage(x, 8, 8)
    x = _fwht_stage(x, 4, 16)
    x = _fwht_stage(x, 2, 32)
    x = _fwht_stage(x, 1, 64)
    return x * 0.08838834764831845


@triton.jit
def _quantize_four_mxfp4_blocks(x):
    """Quantize one 128-wide vector to 64 packed bytes and four scales."""
    x4 = tl.reshape(x, (4, 32))
    amax = tl.max(tl.abs(x4), axis=1)
    amax = tl.maximum(amax, 6.0 * (2.0**-126))
    exponent = tl.math.ceil(tl.math.log2(amax * (1.0 / 6.0)))
    exponent = tl.minimum(tl.maximum(exponent, -127.0), 127.0)
    scales = tl.math.exp2(exponent)
    ue8m0 = (exponent + 127.0).to(tl.uint8)

    pairs = tl.arange(0, 64)
    lo_dim = pairs * 2
    hi_dim = lo_dim + 1
    pair_scale = tl.gather(scales, pairs // 16, axis=0)
    packed = _fp32x2_to_fp4x2(
        tl.gather(x, lo_dim, axis=0) / pair_scale,
        tl.gather(x, hi_dim, axis=0) / pair_scale,
    )
    return packed, ue8m0


@triton.jit
def _fwht_quant_mxfp4_kernel(
    source,
    packed_out,
    scale_out,
    rows,
    source_stride,
    packed_stride,
    scale_stride,
):
    row = tl.program_id(0)
    dims = tl.arange(0, 128)
    x = tl.load(
        source + row * source_stride + dims,
        mask=row < rows,
        other=0.0,
    ).to(tl.float32)
    # Match the FP8 kpool path: the transform output is rounded through BF16
    # before the cache quantizer observes it.
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)
    packed, scales = _quantize_four_mxfp4_blocks(x)
    tl.store(
        packed_out + row * packed_stride + tl.arange(0, 64),
        packed,
        mask=row < rows,
    )
    tl.store(
        scale_out + row * scale_stride + tl.arange(0, 4),
        scales,
        mask=row < rows,
    )


def fwht128_quant_mxfp4(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hadamard-rotate BF16 rows and emit packed E2M1 + UE8M0 scales."""
    if q.ndim != 2 or q.shape[1] != HEAD_DIM or q.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 [rows,{HEAD_DIM}], got {q.dtype} {q.shape}")
    q = q.contiguous()
    rows = int(q.shape[0])
    packed = torch.empty((rows, PACKED_BYTES), dtype=torch.uint8, device=q.device)
    scales = torch.empty((rows, SCALE_BYTES), dtype=torch.uint8, device=q.device)
    if rows:
        _fwht_quant_mxfp4_kernel[(rows,)](
            q,
            packed,
            scales,
            rows,
            q.stride(0),
            packed.stride(0),
            scales.stride(0),
            num_warps=1,
        )
    return packed, scales


@triton.jit
def _kpool_compress_write_mxfp4_kernel(
    cache,
    keys,
    scores,
    ape,
    locations,
    write_mask,
    key_stride_row,
    key_stride_slot,
    score_stride_row,
    score_stride_slot,
    ape_stride_slot,
    page_bytes: tl.constexpr,
    page_size: tl.constexpr,
    pool_size: tl.constexpr,
    has_write_mask: tl.constexpr,
):
    row = tl.program_id(0)
    do_write = True
    if has_write_mask:
        do_write = tl.load(write_mask + row)
    dims = tl.arange(0, 128)
    mask = do_write

    max_score = tl.full((128,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, pool_size):
        score = tl.load(
            scores + row * score_stride_row + slot * score_stride_slot + dims,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(
            ape + slot * ape_stride_slot + dims, mask=mask, other=0.0
        ).to(tl.float32)
        max_score = tl.maximum(max_score, score)

    acc = tl.zeros((128,), tl.float32)
    denom = tl.zeros((128,), tl.float32)
    for slot in tl.static_range(0, pool_size):
        score = tl.load(
            scores + row * score_stride_row + slot * score_stride_slot + dims,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(
            ape + slot * ape_stride_slot + dims, mask=mask, other=0.0
        ).to(tl.float32)
        probability = tl.exp(score - max_score)
        key = tl.load(
            keys + row * key_stride_row + slot * key_stride_slot + dims,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += key * probability
        denom += probability

    x = tl.where(do_write, acc / denom, 0.0).to(tl.bfloat16).to(tl.float32)
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)
    packed, scales = _quantize_four_mxfp4_blocks(x)

    location = tl.load(locations + row, mask=do_write, other=0).to(tl.int64)
    page = location // page_size
    token = location - page * page_size
    page_base = page * page_bytes
    value_record = page_base + token * 64
    scale_record = page_base + page_size * 64 + token * 4
    tl.store(
        cache + value_record + tl.arange(0, 64), packed, mask=do_write
    )
    tl.store(
        cache + scale_record + tl.arange(0, 4),
        scales,
        mask=do_write,
    )


def kpool_compress_and_write_cache_mxfp4(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    *,
    write_mask: torch.Tensor | None = None,
) -> None:
    """Pool, rotate, and write compact MXFP4 records directly to the cache."""
    if kv_cache.dtype != torch.uint8 or kv_cache.ndim != 3:
        raise ValueError(f"expected uint8 rank-3 cache, got {kv_cache.shape}")
    if kv_cache.shape[-1] != RECORD_BYTES:
        raise ValueError(f"expected {RECORD_BYTES}-byte records, got {kv_cache.shape}")
    if slot_k.ndim != 3 or slot_k.shape[-1] != HEAD_DIM:
        raise ValueError(f"unexpected key shape {slot_k.shape}")
    if slot_score.shape != slot_k.shape or ape.shape != slot_k.shape[1:]:
        raise ValueError("kpool score/APE shape mismatch")
    if slot_k.dtype != torch.bfloat16 or ape.dtype != torch.float32:
        raise ValueError("kpool MXFP4 writer expects BF16 keys and FP32 APE")
    rows = int(slot_k.shape[0])
    if not rows:
        return
    slot_k = slot_k.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    loc = loc.to(torch.int64).contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=slot_k.device)
        has_write_mask = False
    else:
        write_mask = write_mask.contiguous()
        has_write_mask = True
    _kpool_compress_write_mxfp4_kernel[(rows,)](
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        slot_k.stride(0),
        slot_k.stride(1),
        slot_score.stride(0),
        slot_score.stride(1),
        ape.stride(0),
        page_bytes=int(kv_cache.stride(0)),
        page_size=int(kv_cache.shape[1]),
        pool_size=int(slot_k.shape[1]),
        has_write_mask=has_write_mask,
        num_warps=1,
    )


@triton.jit
def _kpool_decode_update_mxfp4_kernel(
    cache,
    tail,
    tail_slots,
    keys,
    key_stride_b,
    key_stride_t,
    scores,
    score_stride_b,
    score_stride_t,
    ape,
    ape_stride_slot,
    cache_slots,
    positions,
    next_n,
    page_bytes: tl.constexpr,
    page_size: tl.constexpr,
    pool_size: tl.constexpr,
    tail_stride_block: tl.constexpr,
    tail_stride_half: tl.constexpr,
    tail_stride_slot: tl.constexpr,
):
    """Update one request's tail in position order and write completed pools."""
    request = tl.program_id(0)
    dims = tl.arange(0, 128)

    # The loop must remain ordered: later speculative tokens can consume tail
    # entries written by earlier tokens in the same verification step.
    for token_index in tl.range(0, next_n):
        flat = request * next_n + token_index
        cache_location = tl.load(cache_slots + flat)
        position = tl.load(positions + flat)
        tail_location = tl.load(tail_slots + flat)
        valid_tail = (position >= 0) & (tail_location >= 0)
        safe_position = tl.maximum(position, 0)
        ring_slot = safe_position % pool_size
        tail_block = tl.maximum(tail_location, 0).to(tl.int64) // pool_size
        tail_base = tail_block * tail_stride_block

        key = tl.load(
            keys
            + request * key_stride_b
            + token_index * key_stride_t
            + dims,
            mask=valid_tail,
            other=0.0,
        ).to(tl.float32)
        current_score = tl.load(
            scores
            + request * score_stride_b
            + token_index * score_stride_t
            + dims,
            mask=valid_tail,
            other=0.0,
        ).to(tl.float32)

        completes_pool = (
            valid_tail
            & (cache_location >= 0)
            & (ring_slot == pool_size - 1)
        )
        if completes_pool:
            logical_start = safe_position - ring_slot
            max_score = tl.full((128,), -float("inf"), tl.float32)
            for pool_slot in tl.static_range(0, pool_size):
                is_current = pool_slot == ring_slot
                physical = (logical_start + pool_slot) % pool_size
                saved_score = tl.load(
                    tail
                    + tail_base
                    + tail_stride_half
                    + physical * tail_stride_slot
                    + dims
                ).to(tl.float32)
                score = tl.where(is_current, current_score, saved_score)
                score += tl.load(
                    ape + pool_slot * ape_stride_slot + dims
                ).to(tl.float32)
                max_score = tl.maximum(max_score, score)

            accumulator = tl.zeros((128,), tl.float32)
            denominator = tl.zeros((128,), tl.float32)
            for pool_slot in tl.static_range(0, pool_size):
                is_current = pool_slot == ring_slot
                physical = (logical_start + pool_slot) % pool_size
                saved_score = tl.load(
                    tail
                    + tail_base
                    + tail_stride_half
                    + physical * tail_stride_slot
                    + dims
                ).to(tl.float32)
                score = tl.where(is_current, current_score, saved_score)
                score += tl.load(
                    ape + pool_slot * ape_stride_slot + dims
                ).to(tl.float32)
                probability = tl.exp(score - max_score)
                saved_key = tl.load(
                    tail
                    + tail_base
                    + physical * tail_stride_slot
                    + dims
                ).to(tl.float32)
                pooled_key = tl.where(is_current, key, saved_key)
                accumulator += pooled_key * probability
                denominator += probability

            x = (accumulator / denominator).to(tl.bfloat16).to(tl.float32)
            x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)
            packed, scales = _quantize_four_mxfp4_blocks(x)
            location = cache_location.to(tl.int64)
            page = location // page_size
            token = location - page * page_size
            page_base = page * page_bytes
            value_record = page_base + token * 64
            scale_record = page_base + page_size * 64 + token * 4
            tl.store(cache + value_record + tl.arange(0, 64), packed)
            tl.store(cache + scale_record + tl.arange(0, 4), scales)

        # Preserve rollback/MTP semantics: every real token is stashed, but a
        # completion reads the old ring before overwriting the current slot.
        tl.store(
            tail + tail_base + ring_slot * tail_stride_slot + dims,
            key,
            mask=valid_tail,
        )
        tl.store(
            tail
            + tail_base
            + tail_stride_half
            + ring_slot * tail_stride_slot
            + dims,
            current_score,
            mask=valid_tail,
        )


def kpool_decode_update_and_maybe_write_cache_mxfp4(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
) -> None:
    """SM121 MXFP4 equivalent of vLLM's ordered batched kpool updater."""
    if kv_cache.dtype != torch.uint8 or kv_cache.shape[-1] != RECORD_BYTES:
        raise ValueError(f"expected compact MXFP4 cache, got {kv_cache.shape}")
    if tail_kv_cache.dtype != torch.bfloat16 or tail_kv_cache.ndim != 4:
        raise ValueError(f"unexpected kpool tail cache {tail_kv_cache.shape}")
    if key.ndim != 3 or key.shape[-1] != HEAD_DIM or slot_score.shape != key.shape:
        raise ValueError("unexpected decode key/score geometry")
    requests, next_n = map(int, key.shape[:2])
    if not requests or not next_n:
        return
    expected_grid = (requests, next_n)
    if (
        tail_slot_mapping.shape != expected_grid
        or slot_mapping.shape != expected_grid
        or positions.shape != expected_grid
    ):
        raise ValueError("decode slot/position grid mismatch")
    if tail_kv_cache.shape[1:] != (2, pool_size, HEAD_DIM):
        raise ValueError("tail cache does not match the kpool geometry")
    key = key.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    tail_slot_mapping = tail_slot_mapping.contiguous()
    slot_mapping = slot_mapping.contiguous()
    positions = positions.contiguous()
    _kpool_decode_update_mxfp4_kernel[(requests,)](
        kv_cache,
        tail_kv_cache,
        tail_slot_mapping,
        key,
        key.stride(0),
        key.stride(1),
        slot_score,
        slot_score.stride(0),
        slot_score.stride(1),
        ape,
        ape.stride(0),
        slot_mapping,
        positions,
        next_n,
        page_bytes=int(kv_cache.stride(0)),
        page_size=int(kv_cache.shape[1]),
        pool_size=int(pool_size),
        tail_stride_block=int(tail_kv_cache.stride(0)),
        tail_stride_half=int(tail_kv_cache.stride(1)),
        tail_stride_slot=int(tail_kv_cache.stride(2)),
        num_warps=1,
    )


def _output_buffer(
    kind: str, device: torch.device, rows: int, width: int
) -> torch.Tensor:
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (kind, index, rows, width)
    output = _OUTPUTS.get(key)
    if output is None:
        with _OUTPUTS_LOCK:
            output = _OUTPUTS.get(key)
            if output is None:
                output = torch.empty(
                    (rows, width), dtype=torch.float32, device=device
                )
                _OUTPUTS[key] = output
    return output


@triton.jit
def _paged_mqa_logits_mxfp4_kernel(
    q,
    q_scales,
    cache,
    weights,
    context_lens,
    block_tables,
    indices,
    output,
    q_stride_b: tl.constexpr,
    q_stride_n: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_p: tl.constexpr,
    qs_stride_b: tl.constexpr,
    qs_stride_n: tl.constexpr,
    qs_stride_h: tl.constexpr,
    cache_stride_page: tl.constexpr,
    cache_value_stride: tl.constexpr,
    w_stride_row: tl.constexpr,
    w_stride_h: tl.constexpr,
    lens_stride: tl.constexpr,
    bt_stride_b: tl.constexpr,
    bt_stride_page: tl.constexpr,
    out_stride_row: tl.constexpr,
    out_stride_token: tl.constexpr,
    has_indices: tl.constexpr,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    page_size: tl.constexpr,
    max_model_len: tl.constexpr,
    block_tokens: tl.constexpr,
):
    row = tl.program_id(0)
    tokens = tl.program_id(1) * block_tokens + tl.arange(0, block_tokens)
    output_mask = tokens < max_model_len
    context_len = tl.load(context_lens + row * lens_stride)
    live = output_mask & (tokens < context_len)

    batch = row // next_n
    draft = row - batch * next_n
    table_batch = tl.load(indices + row) if has_indices else batch
    logical_pages = tokens // page_size
    slots = tokens - logical_pages * page_size
    physical_pages = tl.load(
        block_tables + table_batch * bt_stride_b + logical_pages * bt_stride_page,
        mask=live,
        other=0,
    )

    heads = tl.arange(0, num_heads)
    dims = tl.arange(0, 128)
    q_bytes = tl.load(
        q
        + batch * q_stride_b
        + draft * q_stride_n
        + heads[:, None] * q_stride_h
        + (dims[None, :] // 2) * q_stride_p
    )
    q_values = _unpack_e2m1(q_bytes, dims[None, :] & 1)
    q_scale_bytes = tl.load(
        q_scales
        + batch * qs_stride_b
        + draft * qs_stride_n
        + heads[:, None] * qs_stride_h
        + dims[None, :] // 32
    )
    q_values *= tl.math.exp2(q_scale_bytes.to(tl.float32) - 127.0)

    page_bases = physical_pages * cache_stride_page
    value_bases = page_bases + slots * cache_value_stride
    scale_bases = (
        page_bases
        + page_size * cache_value_stride
        + slots * (cache_value_stride // 16)
    )
    k_bytes = tl.load(
        cache + value_bases[None, :] + (dims[:, None] // 2),
        mask=live[None, :],
        other=0,
    )
    k_values = _unpack_e2m1(k_bytes, dims[:, None] & 1)
    k_scale_bytes = tl.load(
        cache + scale_bases[None, :] + dims[:, None] // 32,
        mask=live[None, :],
        other=0,
    )
    k_values *= tl.math.exp2(k_scale_bytes.to(tl.float32) - 127.0)

    scores = tl.dot(
        q_values.to(tl.float16),
        k_values.to(tl.float16),
        out_dtype=tl.float32,
    )
    head_weights = tl.load(
        weights + row * w_stride_row + heads * w_stride_h
    ).to(tl.float32)
    logits = tl.sum(tl.maximum(scores, 0.0) * head_weights[:, None], axis=0)
    logits = tl.where(live, logits, float("-inf"))
    tl.store(
        output + row * out_stride_row + tokens * out_stride_token,
        logits,
        mask=output_mask,
    )


def _normalize_query_scales(
    q_scales: torch.Tensor, batch: int, next_n: int, heads: int
) -> torch.Tensor:
    """Return byte-addressable [B,N,H,4] UE8M0 scales."""
    if q_scales.dtype == torch.int32:
        if q_scales.numel() != batch * next_n * heads:
            raise ValueError(f"unexpected packed Q-scale shape {q_scales.shape}")
        return q_scales.contiguous().view(torch.uint8).reshape(
            batch, next_n, heads, SCALE_BYTES
        )
    if q_scales.dtype == torch.uint8 and q_scales.shape == (
        batch,
        next_n,
        heads,
        SCALE_BYTES,
    ):
        return q_scales
    raise ValueError(f"unexpected Q-scale dtype/shape {q_scales.dtype} {q_scales.shape}")


def paged_mqa_logits_mxfp4(
    q: tuple[torch.Tensor, torch.Tensor],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    *,
    output: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score paged MXFP4 Q/K directly into vLLM's FP32 logits buffer."""
    q_values, q_scales = q
    if q_values.dtype == torch.int8:
        q_values = q_values.view(torch.uint8)
    if q_values.dtype != torch.uint8 or q_values.ndim != 4:
        raise ValueError(f"expected packed Q [B,N,H,64], got {q_values.shape}")
    batch, next_n, heads, packed = map(int, q_values.shape)
    if packed != PACKED_BYTES or heads % 16:
        raise ValueError(f"unexpected packed Q geometry {q_values.shape}")
    q_scales = _normalize_query_scales(q_scales, batch, next_n, heads)
    if kv_cache.dtype != torch.uint8 or kv_cache.ndim != 4:
        raise ValueError(f"expected uint8 rank-4 cache, got {kv_cache.shape}")
    if kv_cache.shape[2:] != (1, RECORD_BYTES):
        raise ValueError(f"unexpected MXFP4 cache shape {kv_cache.shape}")
    rows = batch * next_n
    if weights.shape != (rows, heads):
        raise ValueError(f"unexpected index weights shape {weights.shape}")
    lens = context_lens.reshape(-1)
    if lens.numel() == batch:
        lens = lens[:, None].expand(batch, next_n).reshape(-1)
    elif lens.numel() != rows:
        raise ValueError(f"unexpected context lengths {context_lens.shape}")
    if block_tables.stride(1) != 1:
        block_tables = block_tables.contiguous()
    has_indices = indices is not None
    if indices is None:
        indices = lens
    else:
        indices = indices.reshape(-1)
    if output is None:
        output = _output_buffer("paged", q_values.device, rows, int(max_model_len))
    block_tokens = 64
    _paged_mqa_logits_mxfp4_kernel[
        (rows, triton.cdiv(max_model_len, block_tokens))
    ](
        q_values,
        q_scales,
        kv_cache,
        weights,
        lens,
        block_tables,
        indices,
        output,
        *q_values.stride(),
        *q_scales.stride()[:3],
        kv_cache.stride(0),
        PACKED_BYTES,
        *weights.stride(),
        lens.stride(0),
        *block_tables.stride(),
        *output.stride(),
        has_indices=has_indices,
        next_n=next_n,
        num_heads=heads,
        page_size=int(kv_cache.shape[1]),
        max_model_len=int(max_model_len),
        block_tokens=block_tokens,
        num_warps=4,
        num_stages=2,
    )
    return output


@triton.jit
def _contiguous_mqa_logits_mxfp4_kernel(
    q,
    q_scales,
    k_values,
    k_scales,
    weights,
    starts,
    ends,
    output,
    q_stride_row: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_p: tl.constexpr,
    qs_stride_row: tl.constexpr,
    qs_stride_h: tl.constexpr,
    kv_stride_row: tl.constexpr,
    ks_stride_row: tl.constexpr,
    w_stride_row: tl.constexpr,
    w_stride_h: tl.constexpr,
    out_stride_row: tl.constexpr,
    out_stride_token: tl.constexpr,
    num_heads: tl.constexpr,
    total_k: tl.constexpr,
    block_tokens: tl.constexpr,
):
    row = tl.program_id(0)
    tokens = tl.program_id(1) * block_tokens + tl.arange(0, block_tokens)
    output_mask = tokens < total_k
    start = tl.load(starts + row)
    end = tl.load(ends + row)
    live = output_mask & (tokens >= start) & (tokens < end)
    heads = tl.arange(0, num_heads)
    dims = tl.arange(0, 128)

    q_bytes = tl.load(
        q
        + row * q_stride_row
        + heads[:, None] * q_stride_h
        + (dims[None, :] // 2) * q_stride_p
    )
    q_decoded = _unpack_e2m1(q_bytes, dims[None, :] & 1)
    qs = tl.load(
        q_scales
        + row * qs_stride_row
        + heads[:, None] * qs_stride_h
        + dims[None, :] // 32
    )
    q_decoded *= tl.math.exp2(qs.to(tl.float32) - 127.0)

    k_bytes = tl.load(
        k_values
        + tokens[None, :] * kv_stride_row
        + dims[:, None] // 2,
        mask=live[None, :],
        other=0,
    )
    k_decoded = _unpack_e2m1(k_bytes, dims[:, None] & 1)
    ks = tl.load(
        k_scales
        + tokens[None, :] * ks_stride_row
        + dims[:, None] // 32,
        mask=live[None, :],
        other=0,
    )
    k_decoded *= tl.math.exp2(ks.to(tl.float32) - 127.0)
    scores = tl.dot(
        q_decoded.to(tl.float16),
        k_decoded.to(tl.float16),
        out_dtype=tl.float32,
    )
    head_weights = tl.load(
        weights + row * w_stride_row + heads * w_stride_h
    ).to(tl.float32)
    logits = tl.sum(tl.maximum(scores, 0.0) * head_weights[:, None], axis=0)
    tl.store(
        output + row * out_stride_row + tokens * out_stride_token,
        tl.where(live, logits, float("-inf")),
        mask=output_mask,
    )


def mqa_logits_mxfp4(
    q: tuple[torch.Tensor, torch.Tensor],
    k: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused contiguous MXFP4 scoring for the chunked-prefill indexer."""
    q_values, q_scales = q
    k_values, k_scales = k
    if q_values.dtype == torch.int8:
        q_values = q_values.view(torch.uint8)
    if k_values.dtype == torch.int8:
        k_values = k_values.view(torch.uint8)
    if q_values.ndim != 3 or q_values.shape[-1] != PACKED_BYTES:
        raise ValueError(f"unexpected prefill Q shape {q_values.shape}")
    rows, heads = map(int, q_values.shape[:2])
    q_scales = _normalize_query_scales(q_scales, rows, 1, heads).reshape(
        rows, heads, SCALE_BYTES
    )
    if k_values.shape[-1] != PACKED_BYTES or k_scales.shape[-1] != SCALE_BYTES:
        raise ValueError("unexpected gathered MXFP4 K geometry")
    total_k = int(k_values.shape[0])
    if output is None:
        output = _output_buffer("prefill", q_values.device, rows, total_k)
    block_tokens = 64
    _contiguous_mqa_logits_mxfp4_kernel[
        (rows, triton.cdiv(total_k, block_tokens))
    ](
        q_values,
        q_scales,
        k_values,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        output,
        *q_values.stride(),
        *q_scales.stride()[:2],
        k_values.stride(0),
        k_scales.stride(0),
        *weights.stride(),
        *output.stride(),
        num_heads=heads,
        total_k=total_k,
        block_tokens=block_tokens,
        num_warps=4,
        num_stages=2,
    )
    return output


def clear_output_cache() -> None:
    with _OUTPUTS_LOCK:
        _OUTPUTS.clear()


@triton.jit
def _compressed_pool_slot_mapping_dcp_kernel(
    slot_mapping,
    query_start_loc,
    seq_lens,
    block_table,
    block_table_stride,
    storage_block_size,
    dcp_world_size: tl.constexpr,
    dcp_rank: tl.constexpr,
    compress_ratio: tl.constexpr,
    pad_id: tl.constexpr,
    block_tokens: tl.constexpr,
):
    """Map whole compressed pools to ranks instead of sharding pool members."""
    req = tl.program_id(0)
    query_start = tl.load(query_start_loc + req)
    query_end = tl.load(query_start_loc + req + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens + req)
    first_pos = seq_len - query_len
    virtual_block_size = storage_block_size * dcp_world_size

    for base in range(0, query_len, block_tokens):
        offsets = base + tl.arange(0, block_tokens)
        live = offsets < query_len
        pos = first_pos + offsets
        complete = (pos + 1) % compress_ratio == 0
        pool_pos = pos // compress_ratio
        virtual_offset = pool_pos % virtual_block_size
        owner = virtual_offset % dcp_world_size
        is_local = owner == dcp_rank
        local_offset = virtual_offset // dcp_world_size
        block_index = pool_pos // virtual_block_size
        block_number = tl.load(
            block_table + req * block_table_stride + block_index,
            mask=live & complete & is_local,
            other=0,
        ).to(tl.int64)
        slot = block_number * storage_block_size + local_offset
        slot = tl.where(complete & is_local, slot, pad_id)
        tl.store(slot_mapping + query_start + offsets, slot, mask=live)


def get_compressed_pool_slot_mapping_dcp(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    storage_block_size: int,
    compress_ratio: int,
    dcp_world_size: int,
    dcp_rank: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Return write slots for a cache sharded at compressed-pool granularity.

    Every rank sees all current K/gate inputs, so each complete pool can be
    formed redundantly.  Only the rank owning the resulting pool entry writes
    it.  This preserves four-token GLM pooling while the uncompressed MLA cache
    continues to use ordinary token-granular DCP sharding.
    """
    if compress_ratio <= 1:
        raise ValueError("compressed-pool DCP mapping requires compression")
    if not 0 <= dcp_rank < dcp_world_size:
        raise ValueError("invalid DCP rank")
    out.fill_(-1)
    slot_mapping = out[:num_tokens]
    _compressed_pool_slot_mapping_dcp_kernel[(block_table.shape[0],)](
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        storage_block_size,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        compress_ratio=compress_ratio,
        pad_id=-1,
        block_tokens=1024,
    )
    return slot_mapping


__all__ = [
    "PACKED_BYTES",
    "RECORD_BYTES",
    "SCALE_BYTES",
    "clear_output_cache",
    "fwht128_quant_mxfp4",
    "get_compressed_pool_slot_mapping_dcp",
    "kpool_compress_and_write_cache_mxfp4",
    "mqa_logits_mxfp4",
    "paged_mqa_logits_mxfp4",
]
