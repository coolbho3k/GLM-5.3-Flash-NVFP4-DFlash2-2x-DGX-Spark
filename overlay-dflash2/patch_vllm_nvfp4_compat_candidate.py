#!/usr/bin/env python3
"""Switch the composed GLM candidate from native 288B to compatible 368B MLA."""

from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")


def replace(path: str, old: str, new: str) -> None:
    target = ROOT / path
    source = target.read_text()
    if new in source:
        return
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"5A compatibility anchor count {count} in {target}")
    target.write_text(source.replace(old, new, 1))


replace(
    "v1/kv_cache_interface.py",
    '''            if self.head_size == 512 and self.model_version != "deepseek_v4":
                return self.block_size * 288
''',
    '''            if self.head_size == 512 and self.model_version != "deepseek_v4":
                # Stage 5A compatibility ABI: 288B NVFP4 latent plus the
                # retained 80B FP8-RoPE field.
                return self.block_size * 368
''',
)

replace(
    "v1/attention/backends/mla/flashinfer_mla_sparse.py",
    '''            if head_size == 512:
                return (num_blocks, block_size, 288)
''',
    '''            if head_size == 512:
                return (num_blocks, block_size, 368)
''',
)

replace(
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
    '''            out, lse = run_unified_prefill(
                q=q,
                kv_cache=kv_c_and_k_pe_cache.view(torch.uint8),
                topk_indices=topk_indices_physical,
                topk_length=kernel_topk_lens,
                sm_scale=self.scale,
                page_block_size=attn_metadata.block_size,
                output=output,
                scale_format=ScaleFormat.NVFP4_E4M3,
                fp8_rope=False,
            )
''',
    '''            # GLM has no RoPE payload, but Stage 5A deliberately keeps
            # the generic 368-byte ABI. Feed an explicit zero query field so
            # the compatibility reader follows the same numerical path.
            q_compatible = torch.nn.functional.pad(q, (0, 64))
            out, lse = run_unified_prefill(
                q=q_compatible,
                kv_cache=kv_c_and_k_pe_cache.view(torch.uint8),
                topk_indices=topk_indices_physical,
                topk_length=kernel_topk_lens,
                sm_scale=self.scale,
                page_block_size=attn_metadata.block_size,
                output=output,
                scale_format=ScaleFormat.NVFP4_E4M3,
                fp8_rope=True,
            )
''',
)

replace(
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
    '''            from b12x.attention._shared.mla.kv_cache import (
                concat_and_cache_nvfp4_mla_zero_rope,
            )
            concat_and_cache_nvfp4_mla_zero_rope(
                kv_c_normed,
                kv_cache.view(torch.uint8),
                slot_mapping.flatten(),
                scale=k_scale,
            )
''',
    '''            from b12x.attention._shared.mla.kv_cache import (
                concat_and_cache_nvfp4_mla_fp8_rope,
            )
            # Preserve the compatibility field while retaining GLM's exact
            # zero-RoPE semantics. This is a rollback/capacity control, not the
            # fastest production path; Stage 5B removes this allocation/write.
            compatibility_rope = kv_c_normed.new_zeros(
                (kv_c_normed.shape[0], 64)
            )
            concat_and_cache_nvfp4_mla_fp8_rope(
                kv_c_normed,
                compatibility_rope,
                kv_cache.view(torch.uint8),
                slot_mapping.flatten(),
                scale=k_scale,
            )
''',
)

replace(
    "platforms/interface.py",
    '''                attn_page_size_1_token = 288
''',
    '''                attn_page_size_1_token = 368
''',
)

# The compatible MLA record is 368 = 23 * 16 bytes while one DFlash token is
# 1024 bytes. Exact page sharing therefore gives
#
#     dflash_block = mla_block * 368 / 1024 = mla_block * 23 / 64.
#
# FlashInfer requires the resulting DFlash block to be divisible by 32, so the
# MLA block must be divisible by 2048. Retaining the native layout's 1024-token
# alignment selects a 5120-token MLA block and a non-integral 1840-token DFlash
# block; that falls back to standalone drafter tensors and makes the 65K
# admission requirement exceed available memory. A 2048-token alignment
# selects 6144/2208-token MLA/DFlash pages and preserves exact sharing.
replace(
    "platforms/interface.py",
    '''                dflash_exact_fit_alignment = 1024
''',
    '''                dflash_exact_fit_alignment = 2048
''',
)
