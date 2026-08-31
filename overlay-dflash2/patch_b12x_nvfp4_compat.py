#!/usr/bin/env python3
"""Fix the 368-byte NVFP4 compatibility record stride in B12X prefill."""

from pathlib import Path


PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "b12x/attention/_shared/mla/io_mg.py"
)

source = PATH.read_text()
old = '''    if cutlass.const_expr(scale_format == 2):
        if cutlass.const_expr(kv_smem_stride == 288):
            _ios = Int64(288)
        elif cutlass.const_expr(fp8_rope):
            _ios = Int64(_NVFP4_FP8_ROPE_IO_STRIDE)
        else:
            _ios = Int64(_NVFP4_IO_STRIDE)
'''
new = '''    if cutlass.const_expr(scale_format == 2):
        # Both NVFP4 layouts stage the same 288-byte latent in shared memory.
        # Select the explicit 368-byte FP8-RoPE global ABI first; otherwise the
        # compatibility path advances between records at 288 bytes and reads
        # every token after the first from a misaligned address.
        if cutlass.const_expr(fp8_rope):
            _ios = Int64(_NVFP4_FP8_ROPE_IO_STRIDE)
        elif cutlass.const_expr(kv_smem_stride == 288):
            _ios = Int64(288)
        else:
            _ios = Int64(_NVFP4_IO_STRIDE)
'''

if new in source:
    print(f"nvfp4_compat: {PATH} already patched")
elif source.count(old) == 1:
    PATH.write_text(source.replace(old, new, 1))
    print(f"nvfp4_compat: patched {PATH}")
else:
    raise RuntimeError(
        f"NVFP4 compatibility stride anchor count is {source.count(old)}, expected 1"
    )
