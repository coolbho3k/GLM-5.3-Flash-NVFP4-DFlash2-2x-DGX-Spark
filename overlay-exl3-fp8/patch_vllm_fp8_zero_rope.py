"""Select the native 528-byte GLM NoPE FP8 MLA layout on SM121."""

from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")


def replace(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"FP8 zero-RoPE patch anchor missing in {target}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1))


replace(
    "v1/kv_cache_interface.py",
    """        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":
""",
    """        if self.cache_dtype_str == "fp8_ds_mla":
            if self.head_size == 512 and self.model_version != "deepseek_v4":
                return self.block_size * 528
            if self.model_version == "deepseek_v4":
""",
)

replace(
    "v1/attention/backends/mla/flashinfer_mla_sparse.py",
    """        if cache_dtype_str in ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
            # fp8_ds_mla packed layout: 512 NoPE + 16 scales + 128 RoPE.
            return (num_blocks, block_size, 656)
""",
    """        if cache_dtype_str in ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
            if head_size == 512:
                # GLM-5.3 NoPE: 512 E4M3 values + four FP32 group scales.
                return (num_blocks, block_size, 528)
            # fp8_ds_mla packed layout: 512 NoPE + 16 scales + 128 RoPE.
            return (num_blocks, block_size, 656)
""",
)

replace(
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
    """        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
""",
    """        if self.qk_rope_head_dim == 0:
            from b12x.attention._shared.mla.prefill import run_unified_prefill
            from b12x.attention._shared.mla.traits import ScaleFormat

            out, _ = run_unified_prefill(
                q=q,
                kv_cache=kv_c_and_k_pe_cache.view(torch.uint8),
                topk_indices=topk_indices_physical,
                sm_scale=self.scale,
                page_block_size=attn_metadata.block_size,
                output=output,
                scale_format=ScaleFormat.ARBITRARY_FP32,
                fp8_rope=False,
            )
            return out, None

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
""",
)

replace(
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
    """        return out.squeeze(1), None
""",
    """        return out.squeeze(1), None

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        if self.qk_rope_head_dim == 0:
            from b12x.attention._shared.mla.fp8_zero_rope import (
                concat_and_cache_fp8_mla_zero_rope,
            )

            concat_and_cache_fp8_mla_zero_rope(
                kv_c_normed,
                kv_cache.view(torch.uint8),
                slot_mapping.flatten(),
            )
            return
        super().do_kv_cache_update(
            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
        )
""",
)

replace(
    "platforms/cuda.py",
    """            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]
""",
    """            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
            ]
""",
)

# The DFlash group is allocation-only whether exact-fit or standalone. It must
# not constrain the scheduler block LCM or prefix-hash geometry.
replace(
    "v1/kv_cache_interface.py",
    """class SlidingWindowSpec(AttentionSpec):
    sliding_window: int
    head_size_v: int = None  # type: ignore[assignment]

    def __post_init__(self):
""",
    """class SlidingWindowSpec(AttentionSpec):
    sliding_window: int
    head_size_v: int = None  # type: ignore[assignment]
    exclude_from_prefix_caching: bool = False

    @property
    def participates_in_prefix_caching(self) -> bool:
        return not self.exclude_from_prefix_caching

    def __post_init__(self):
""",
)

replace(
    "v1/core/kv_cache_utils.py",
    """            # A 64-divisible manager block is divisible by every int kernel
            # block size the SWA backends register (16/32/64), so
            # select_common_block_size always finds a clean split.
            and fit_block % 64 == 0
            # Keep resolve_kv_cache_block_sizes' scheduler LCM at
            # max(mla_block, fit_block) instead of exploding.
            and (fit_block % mla_block == 0 or mla_block % fit_block == 0)
            and len(draft_specs) <= len(mla_names)
""",
    """            # FlashInfer can split this non-prefix draft page at 32-token
            # kernel granularity. It need not divide the main scheduler block.
            and fit_block % 32 == 0
            and len(draft_specs) <= len(mla_names)
""",
)

replace(
    "v1/core/kv_cache_utils.py",
    """            new_draft_specs = dict(draft_specs)
""",
    """            # Padded DFlash slot-share: manager block 64 matches the SWA
            # kernel block, so the strided view has exactly one kernel page per
            # physical MLA page. This safely avoids standalone draft tensors.
            compact_block = 64
            logger.info(
                "DFlash2 drafter KV: padded slot-share block=%d mla_page=%d "
                "(was block=%d); draft_bytes/token=%d",
                compact_block,
                mla_page,
                any_draft.block_size,
                draft_bytes_per_token,
            )
            new_draft_specs = {
                name: replace(
                    s,
                    block_size=compact_block,
                    page_size_padded=mla_page,
                    exclude_from_prefix_caching=True,
                )
                for name, s in draft_specs.items()
            }
""",
)

replace(
    "v1/core/kv_cache_utils.py",
    """        if any(s.page_size_padded is not None for s in draft_inner.values()):
            return None
""",
    """        if any(s.page_size_padded is not None for s in draft_inner.values()):
            # A padded draft view is safe only when its manager page is also
            # the 64-token FlashAttention kernel page.
            if any(
                s.block_size != 64 or s.page_size_padded != mla_page
                for s in draft_inner.values()
            ):
                return None
""",
)

replace(
    "v1/core/kv_cache_utils.py",
    """                name: replace(s, block_size=fit_block)
                for name, s in draft_specs.items()
""",
    """                name: replace(
                    s,
                    block_size=fit_block,
                    exclude_from_prefix_caching=True,
                )
                for name, s in draft_specs.items()
""",
)

replace(
    "v1/core/kv_cache_utils.py",
    """    scheduler_block_size = math.lcm(*group_block_sizes)
""",
    """    scheduling_sizes = [
        bs
        for g, bs in zip(groups, group_block_sizes)
        if g.kv_cache_spec.participates_in_prefix_caching
    ] or group_block_sizes
    scheduler_block_size = math.lcm(*scheduling_sizes)
""",
)

replace(
    "v1/core/kv_cache_coordinator.py",
    """        assert scheduler_block_size % hash_block_size == 0 and all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_config.kv_cache_groups
        )
""",
    """        assert scheduler_block_size % hash_block_size == 0 and all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_config.kv_cache_groups
            if g.kv_cache_spec.participates_in_prefix_caching
        )
""",
)

replace(
    "v1/core/single_type_kv_cache_manager.py",
    """        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size
""",
    """        if not self.kv_cache_spec.participates_in_prefix_caching:
            return

        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size
""",
)

# Preserve the target and draft cache dtypes independently.
replace(
    "model_executor/layers/attention/mla_attention.py",
    """        if normalized_kv_cache_dtype != kv_cache_dtype:
            if cache_config is not None:
                cache_config.cache_dtype = normalized_kv_cache_dtype
            kv_cache_dtype = normalized_kv_cache_dtype
""",
    """        if normalized_kv_cache_dtype != kv_cache_dtype:
            kv_cache_dtype = normalized_kv_cache_dtype
""",
)

replace(
    "platforms/interface.py",
    """            attn_page_size_1_token = MLAAttentionSpec(
                block_size=1,
                num_kv_heads=model_config.get_num_kv_heads(parallel_config),
                head_size=model_config.get_head_size(),
                dtype=kv_cache_dtype,
                cache_dtype_str=cache_config.cache_dtype,
                kv_quant_mode=kv_quant_mode,
            ).page_size_bytes
""",
    """            attn_page_size_1_token = MLAAttentionSpec(
                block_size=1,
                num_kv_heads=model_config.get_num_kv_heads(parallel_config),
                head_size=model_config.get_head_size(),
                dtype=kv_cache_dtype,
                cache_dtype_str=cache_config.cache_dtype,
                kv_quant_mode=kv_quant_mode,
            ).page_size_bytes
            if (
                model_config.get_head_size() == 512
                and getattr(model_config.hf_config, "qk_rope_head_dim", None) == 0
                and cache_config.cache_dtype in ("fp8", "fp8_e4m3")
            ):
                attn_page_size_1_token = 528
""",
)

replace(
    "model_executor/layers/attention/mla_attention.py",
    """            cache_dtype_str=vllm_config.cache_config.cache_dtype,
""",
    """            cache_dtype_str=self.kv_cache_dtype,
""",
)
