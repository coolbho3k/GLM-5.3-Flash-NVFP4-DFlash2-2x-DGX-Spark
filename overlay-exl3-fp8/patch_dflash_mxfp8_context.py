#!/usr/bin/env python3
"""Allow ModelOpt MXFP8 QKV weights in DFlash's fused context-KV builder.

The normal draft-token QKV path stays quantized and runs through B12X.  The
context precomputation path concatenates K/V rows and calls torch F.linear,
so it needs a one-time BF16 materialization of those rows after loading.
"""

from pathlib import Path


PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/models/qwen3_dflash.py"
)


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one DFlash context anchor, found {count}")
    return source.replace(old, new, 1)


source = PATH.read_text()
old = """        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
"""
new = """        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        # ModelOpt MXFP8 linears retain E4M3 values plus E8M0 block scales.
        # Draft-token QKV remains packed/quantized, but this fused F.linear path
        # needs a BF16 materialization.  Keep only the K/V rows long-term.
        kv_weights = []
        for attention in layers_attn:
            qkv_weight = attention.qkv_proj.weight
            if qkv_weight.dtype == torch.float8_e4m3fn:
                weight_scale = getattr(attention.qkv_proj, "weight_scale", None)
                if weight_scale is None:
                    raise RuntimeError("MXFP8 DFlash QKV is missing weight_scale")
                from b12x.gemm._shared.wo_mxfp8 import (
                    dequantize_mxfp8_rows_torch,
                )

                scale_rows = weight_scale.data
                if scale_rows.ndim == 2:
                    scale_rows = scale_rows.unsqueeze(0)
                qkv_weight = dequantize_mxfp8_rows_torch(
                    qkv_weight.data, scale_rows
                ).to(dtype=self._hidden_norm_weight.dtype)
            kv_weights.append(qkv_weight[attention.q_size :])
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
"""

if old in source:
    source = replace_once(source, old, new)
elif "dequantize_mxfp8_rows_torch" not in source:
    raise RuntimeError("DFlash MXFP8 context patch is neither applicable nor present")

PATH.write_text(source)
