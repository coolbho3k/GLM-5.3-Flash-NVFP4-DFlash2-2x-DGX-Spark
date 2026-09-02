#!/usr/bin/env python3
"""Add a runtime per-layer outer scale to the 288-byte NVFP4 MLA writer.

The cache ABI is unchanged. For reader outer scale ``L`` the writer encodes
E4M3(group_amax / (divisor * L)); the reader reconstructs E2M1 * E4M3 * L.
The existing four-over-six decision remains local to each group.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "b12x/attention/_shared/mla/kv_cache.py"
)
MARKER = "NVFP4 CALIBRATED_OUTER_SCALE runtime argument"


REPLACEMENTS = [
    (
        "kernel __call__ signature",
        """        slot_capacity: Int32,
        num_tokens: Int32,
        stream: cuda.CUstream,
""",
        """        slot_capacity: Int32,
        num_tokens: Int32,
        latent_scale: Float32,  # NVFP4 CALIBRATED_OUTER_SCALE runtime argument
        stream: cuda.CUstream,
""",
    ),
    (
        "kernel call",
        """            entry_stride,
            slot_capacity,
        ).launch(
""",
        """            entry_stride,
            slot_capacity,
            latent_scale,
        ).launch(
""",
    ),
    (
        "kernel signature",
        """        entry_stride: Int32,
        slot_capacity: Int32,
    ):
""",
        """        entry_stride: Int32,
        slot_capacity: Int32,
        latent_scale: Float32,
    ):
""",
    ),
    (
        "four-over-six comment",
        """                    # NVFP4 FOUR_OVER_SIX scale search at global scale 1.0.
""",
        """                    # NVFP4 FOUR_OVER_SIX with a calibrated reader outer
                    # scale. L=latent_scale and G=1/L; scaling the local E4M3
                    # value by G keeps small group scales out of the zero/subnormal
                    # range without changing the 288-byte record ABI.
""",
    ),
    (
        "six scale expression",
        """                    scale6_u32 = cvt_f32_to_e4m3(
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
""",
        """                    inv_latent = rcp_approx_ftz(latent_scale)
                    scale6_u32 = cvt_f32_to_e4m3(
                        (group_amax * rcp_approx_ftz(Float32(6.0))) * inv_latent
                    )
                    decoded6 = cvt_e4m3_to_f32_via_f16(scale6_u32)
                    inv6 = Float32(0.0)
                    dequant6 = Float32(0.0)
                    packed6 = Uint64(0)
                    if decoded6 != Float32(0.0):
                        inv6 = rcp_approx_ftz(decoded6) * inv_latent
                        dequant6 = rcp_approx_ftz(inv6)
                        packed6 = quantize_and_pack_16_fast(vals, inv6)
""",
    ),
    (
        "four scale expression",
        """                    scale4_u32 = cvt_f32_to_e4m3(
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
""",
        """                    scale4_u32 = cvt_f32_to_e4m3(
                        (group_amax * rcp_approx_ftz(Float32(4.0))) * inv_latent
                    )
                    decoded4 = cvt_e4m3_to_f32_via_f16(scale4_u32)
                    inv4 = Float32(0.0)
                    dequant4 = Float32(0.0)
                    packed4 = Uint64(0)
                    if decoded4 != Float32(0.0):
                        inv4 = rcp_approx_ftz(decoded4) * inv_latent
                        dequant4 = rcp_approx_ftz(inv4)
                        packed4 = quantize_and_pack_16_fast(vals, inv4)
""",
    ),
    (
        "flat-launch signature",
        """    per_token_scale: bool = False,
    zero_rope: bool = False,
) -> None:
""",
        """    per_token_scale: bool = False,
    zero_rope: bool = False,
    latent_scale: float = 1.0,
) -> None:
""",
    ),
    (
        "launch arguments",
        """        Int32(slot_capacity),
        Int32(num_tokens),
        current_cuda_stream(),
""",
        """        Int32(slot_capacity),
        Int32(num_tokens),
        Float32(float(latent_scale)),
        current_cuda_stream(),
""",
    ),
    (
        "compile version and comment",
        """    # Version 5: four-over-six changes writer output while preserving the
    # record ABI.  Never reuse a v4 cubin containing the amax/6-only writer.
    spec = KernelCompileSpec.from_key(
        "attention.mla.nvfp4_fp8_rope_kv_cache",
        5,
""",
        """    # Version 6: the outer scale is now a runtime kernel argument. The
    # record ABI remains unchanged, but a v5 cubin has the wrong signature.
    spec = KernelCompileSpec.from_key(
        "attention.mla.nvfp4_fp8_rope_kv_cache",
        6,
""",
    ),
    (
        "compile labels",
        """            "per_token_scale",
            "zero_rope",
        ),
""",
        """            "per_token_scale",
            "zero_rope",
            "latent_scale",
        ),
""",
    ),
    (
        "zero-rope op signature",
        """def _concat_and_cache_nvfp4_mla_zero_rope_op(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
""",
        """def _concat_and_cache_nvfp4_mla_zero_rope_op(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    latent_scale: float = 1.0,
) -> None:
""",
    ),
    (
        "zero-rope flat launch",
        """        per_token_scale=False,
        zero_rope=True,
    )
""",
        """        per_token_scale=False,
        zero_rope=True,
        latent_scale=latent_scale,
    )
""",
    ),
    (
        "zero-rope fake signature",
        """def _concat_and_cache_nvfp4_mla_zero_rope_fake(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
""",
        """def _concat_and_cache_nvfp4_mla_zero_rope_fake(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    latent_scale: float = 1.0,
) -> None:
""",
    ),
    (
        "public zero-rope signature",
        """def concat_and_cache_nvfp4_mla_zero_rope(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> None:
""",
        """def concat_and_cache_nvfp4_mla_zero_rope(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    scale: torch.Tensor | None = None,
    latent_scale: float = 1.0,
) -> None:
""",
    ),
    (
        "public zero-rope doc",
        """    per-token outer scale; ``scale`` is accepted for cache-op signature parity.
    """,
        """    per-token outer scale; ``scale`` is accepted for cache-op signature parity.
    ``latent_scale`` is the positive per-layer outer scale L applied by both
    writer reconstruction and the attention reader (global G = 1/L).
    """,
    ),
    (
        "public zero-rope validation",
        '''    ``latent_scale`` is the positive per-layer outer scale L applied by both
    writer reconstruction and the attention reader (global G = 1/L).
    """
    del scale
    if kv_c.ndim != 2 or int(kv_c.shape[1]) != _KV_LORA_RANK:
''',
        '''    ``latent_scale`` is the positive per-layer outer scale L applied by both
    writer reconstruction and the attention reader (global G = 1/L).
    """
    del scale
    latent_scale = float(latent_scale)
    if not (latent_scale > 0.0 and latent_scale < float("inf")):
        raise ValueError("latent_scale must be finite and positive")
    if kv_c.ndim != 2 or int(kv_c.shape[1]) != _KV_LORA_RANK:
''',
    ),
    (
        "public zero-rope op call",
        """    torch.ops.b12x.concat_and_cache_nvfp4_mla_zero_rope(
        kv_c, kv_cache, slot_mapping
    )
""",
        """    torch.ops.b12x.concat_and_cache_nvfp4_mla_zero_rope(
        kv_c, kv_cache, slot_mapping, latent_scale
    )
""",
    ),
]


def patch(target: Path) -> bool:
    text = target.read_text()
    if MARKER in text:
        print(f"already patched: {target}")
        return False
    for description, old, new in REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(
                f"expected exactly one {description} anchor in {target}, found {count}"
            )
        text = text.replace(old, new, 1)
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
