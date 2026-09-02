#!/usr/bin/env python3
"""Port vLLM's NVFP4 four-over-six scale search into B12x's MLA writer.

The on-disk/on-cache ABI is unchanged.  Every group still occupies eight
packed E2M1 bytes plus one E4M3 scale byte.  Only the store-time choice of
that scale changes: evaluate amax/6 and amax/4, then retain the candidate
with the lower reconstruction SSE (ties retain the conventional /6 scale).

This patch targets the SparkInfer source pinned by build-production-image.sh
after the repository's native 288-byte GLM zero-RoPE patch has been applied.
It is deliberately strict and idempotent so an upstream source drift cannot
silently produce a partially patched serving image.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "b12x/attention/_shared/mla/kv_cache.py"
)
MARKER = "NVFP4 FOUR_OVER_SIX scale search"


DOC_OLD = """    [ 256, 288)  32 x E4M3 group scale bytes (group amax / 6.0)
"""
DOC_NEW = """    [ 256, 288)  32 x E4M3 group scale bytes (chosen amax/6 or amax/4)
"""


IMPORT_OLD = """    f16x2_to_f32x2,
    fabs_f32,
"""
IMPORT_NEW = """    f16x2_to_f32x2,
    fp4_decode_4bytes,
    fabs_f32,
"""


HELPER_ANCHOR = """    return Float32(f0), Float32(f1)


class ConcatAndCacheNvfp4MlaFp8RopeKernel:
"""


HELPER_REPLACEMENT = """    return Float32(f0), Float32(f1)


@cute.jit
def _nvfp4_packed_reconstruction_error_16(
    vals: cute.Tensor,
    packed: Uint64,
    dequant_scale: Float32,
) -> Float32:
    \"\"\"SSE for the exact 16 E2M1 codes produced by the SM121 converter.\"\"\"
    error = Float32(0.0)

    # Decode with the same native E2M1 instruction family used by the
    # attention reader.  Besides matching cvt.rn.satfinite.e2m1 exactly at
    # tie points, evaluating already-packed candidates lets the winner be
    # written directly instead of running a third FP4 conversion.
    h0, h1, h2, h3 = fp4_decode_4bytes(
        (packed & Uint64(0xFFFFFFFF)).to(Uint32)
    )
    q0, q1 = f16x2_to_f32x2(h0)
    q2, q3 = f16x2_to_f32x2(h1)
    q4, q5 = f16x2_to_f32x2(h2)
    q6, q7 = f16x2_to_f32x2(h3)
    d0 = q0 * dequant_scale - vals[0]
    d1 = q1 * dequant_scale - vals[1]
    d2 = q2 * dequant_scale - vals[2]
    d3 = q3 * dequant_scale - vals[3]
    d4 = q4 * dequant_scale - vals[4]
    d5 = q5 * dequant_scale - vals[5]
    d6 = q6 * dequant_scale - vals[6]
    d7 = q7 * dequant_scale - vals[7]
    error = (
        d0 * d0
        + d1 * d1
        + d2 * d2
        + d3 * d3
        + d4 * d4
        + d5 * d5
        + d6 * d6
        + d7 * d7
    )

    h0, h1, h2, h3 = fp4_decode_4bytes((packed >> Uint64(32)).to(Uint32))
    q0, q1 = f16x2_to_f32x2(h0)
    q2, q3 = f16x2_to_f32x2(h1)
    q4, q5 = f16x2_to_f32x2(h2)
    q6, q7 = f16x2_to_f32x2(h3)
    d0 = q0 * dequant_scale - vals[8]
    d1 = q1 * dequant_scale - vals[9]
    d2 = q2 * dequant_scale - vals[10]
    d3 = q3 * dequant_scale - vals[11]
    d4 = q4 * dequant_scale - vals[12]
    d5 = q5 * dequant_scale - vals[13]
    d6 = q6 * dequant_scale - vals[14]
    d7 = q7 * dequant_scale - vals[15]
    return Float32(
        error
        + d0 * d0
        + d1 * d1
        + d2 * d2
        + d3 * d3
        + d4 * d4
        + d5 * d5
        + d6 * d6
        + d7 * d7
    )


class ConcatAndCacheNvfp4MlaFp8RopeKernel:
"""


WRITER_OLD = """                else:
                    # NVFP4 block quant at global scale 1.0: scale byte =
                    # e4m3(amax/6); values scaled by rcp.approx.ftz of the
                    # hardware-exact decode of that byte (what the reader
                    # multiplies back), then satfinite E2M1.
                    scale_f32 = group_amax * rcp_approx_ftz(Float32(6.0))
                    scale_u32 = cvt_f32_to_e4m3(scale_f32)
                    decoded_scale = cvt_e4m3_to_f32_via_f16(scale_u32)
                    packed64 = Uint64(0)
                    if decoded_scale != Float32(0.0):
                        packed64 = quantize_and_pack_16_fast(
                            vals, rcp_approx_ftz(decoded_scale)
                        )
                    st_global_u64(dst + (tid * Int32(8)).to(Int64), packed64)
                    st_global_u8(
                        dst + Int64(_NOPE_BYTES) + tid.to(Int64),
                        cutlass.Uint8(scale_u32 & Uint32(0xFF)),
                    )
"""


WRITER_NEW = """                else:
                    # NVFP4 FOUR_OVER_SIX scale search at global scale 1.0.
                    # This is the algorithm exposed upstream by vLLM as
                    # nvfp4_4over6: construct E4M3(amax/6) and E4M3(amax/4),
                    # evaluate reconstruction SSE over the 16 actual E2M1
                    # codes, and use /4 only when it is strictly better.
                    # Ties deliberately preserve the conventional /6 result.
                    scale6_u32 = cvt_f32_to_e4m3(
                        group_amax * rcp_approx_ftz(Float32(6.0))
                    )
                    decoded6 = cvt_e4m3_to_f32_via_f16(scale6_u32)
                    inv6 = Float32(0.0)
                    dequant6 = Float32(0.0)
                    packed6 = Uint64(0)
                    if decoded6 != Float32(0.0):
                        inv6 = rcp_approx_ftz(decoded6)
                        dequant6 = rcp_approx_ftz(inv6)
                        packed6 = quantize_and_pack_16_fast(vals, inv6)
                    err6 = _nvfp4_packed_reconstruction_error_16(
                        vals, packed6, dequant6
                    )

                    scale4_u32 = cvt_f32_to_e4m3(
                        group_amax * rcp_approx_ftz(Float32(4.0))
                    )
                    decoded4 = cvt_e4m3_to_f32_via_f16(scale4_u32)
                    inv4 = Float32(0.0)
                    dequant4 = Float32(0.0)
                    packed4 = Uint64(0)
                    if decoded4 != Float32(0.0):
                        inv4 = rcp_approx_ftz(decoded4)
                        dequant4 = rcp_approx_ftz(inv4)
                        packed4 = quantize_and_pack_16_fast(vals, inv4)
                    err4 = _nvfp4_packed_reconstruction_error_16(
                        vals, packed4, dequant4
                    )

                    scale_u32 = scale6_u32
                    packed64 = packed6
                    if err4 < err6:
                        scale_u32 = scale4_u32
                        packed64 = packed4
                    st_global_u64(dst + (tid * Int32(8)).to(Int64), packed64)
                    st_global_u8(
                        dst + Int64(_NOPE_BYTES) + tid.to(Int64),
                        cutlass.Uint8(scale_u32 & Uint32(0xFF)),
                    )
"""


SPEC_OLD = """    # Version 3: per_token_scale joined the record semantics (fp32 second-level
    # scale at [292, 296)); a v2 cubin must never run against v3 records.
    spec = KernelCompileSpec.from_key(
        "attention.mla.nvfp4_fp8_rope_kv_cache",
        4,
"""


SPEC_NEW = """    # Version 5: four-over-six changes writer output while preserving the
    # record ABI.  Never reuse a v4 cubin containing the amax/6-only writer.
    spec = KernelCompileSpec.from_key(
        "attention.mla.nvfp4_fp8_rope_kv_cache",
        5,
"""


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one {description} anchor, found {count}"
        )
    return text.replace(old, new, 1)


def patch(target: Path) -> bool:
    text = target.read_text()
    if MARKER in text:
        print(f"already patched: {target}")
        return False

    text = replace_once(text, DOC_OLD, DOC_NEW, "record-layout documentation")
    text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "intrinsics import")
    text = replace_once(text, HELPER_ANCHOR, HELPER_REPLACEMENT, "helper")
    text = replace_once(text, WRITER_OLD, WRITER_NEW, "static NVFP4 writer")
    text = replace_once(text, SPEC_OLD, SPEC_NEW, "kernel cache version")
    target.write_text(text)
    print(f"patched: {target}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    if not args.target.is_file():
        raise FileNotFoundError(args.target)
    patch(args.target)


if __name__ == "__main__":
    main()
