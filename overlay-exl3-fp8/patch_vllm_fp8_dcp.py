"""Enable GLM-5.3 zero-RoPE FP8 MLA and pooled indexer under TP2/DCP2."""

from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")


def replace(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"DCP patch anchor missing in {target}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1))


replace(
    "v1/attention/backends/mla/indexer.py",
    "from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping\n",
    '''from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from b12x.attention.dsa_indexer.sm121_mxfp4 import (
    get_compressed_pool_slot_mapping_dcp,
)
''',
)

replace(
    "v1/attention/backends/mla/indexer.py",
    '''        if self.dcp_world_size > 1 and self.compress_ratio > 1:
            raise NotImplementedError(
                "DCP is not supported with sparse indexer KV compression "
                f"(compress_ratio={self.compress_ratio})."
            )
''',
    '''        # GLM kpool compression is sharded at complete-pool granularity.
        # The custom slot mapper below assigns each compressed entry to one
        # rank, while the ordinary MLA cache remains token-sharded.
''',
)

replace(
    "v1/attention/backends/mla/indexer.py",
    '''            compressed_slot_mapping = get_compressed_slot_mapping(
                num_tokens,
                query_start_loc,
                seq_lens,
                indexer_block_table,
                self.kv_cache_spec.storage_block_size,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )
''',
    '''            if self.dcp_world_size > 1:
                compressed_slot_mapping = get_compressed_pool_slot_mapping_dcp(
                    num_tokens,
                    query_start_loc,
                    seq_lens,
                    indexer_block_table,
                    self.kv_cache_spec.storage_block_size,
                    self.compress_ratio,
                    self.dcp_world_size,
                    self.dcp_rank,
                    self.compressed_slot_mapping_buffer,
                )
            else:
                compressed_slot_mapping = get_compressed_slot_mapping(
                    num_tokens,
                    query_start_loc,
                    seq_lens,
                    indexer_block_table,
                    self.kv_cache_spec.storage_block_size,
                    self.compress_ratio,
                    out=self.compressed_slot_mapping_buffer,
                )
''',
)

replace(
    "v1/core/kv_cache_coordinator.py",
    '''        if dcp_world_size > 1:
            # DCP shards full-attention KV across ranks and replicates Mamba
            # state; other spec types (e.g. sliding window) have no DCP-aware
            # handling yet, so reject them explicitly.
            for g in kv_cache_config.kv_cache_groups:
                assert isinstance(g.kv_cache_spec, (FullAttentionSpec, MambaSpec)), (
                    "DCP with hybrid KV cache layouts only supports "
                    "full-attention and Mamba groups, got: "
                    f"{type(g.kv_cache_spec).__name__}."
                )
''',
    '''        if dcp_world_size > 1:
            # GLM groups MLA/indexer and DFlash layers in uniform wrappers.
            # Their attention members are DCP-sharded; Mamba remains replicated,
            # and KpoolTailSpec rebuilds a replicated circular slot mapping.
            for g in kv_cache_config.kv_cache_groups:
                spec = g.kv_cache_spec
                inner = getattr(spec, "kv_cache_specs", None)
                supported_uniform = inner is not None and all(
                    isinstance(s, (FullAttentionSpec, SlidingWindowSpec))
                    for s in inner.values()
                )
                assert (
                    isinstance(spec, (FullAttentionSpec, SlidingWindowSpec, MambaSpec))
                    or supported_uniform
                ), (
                    "Unsupported GLM DCP cache group: "
                    f"{type(spec).__name__}."
                )
''',
)

replace(
    "v1/kv_cache_interface.py",
    '''    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        assert vllm_config.parallel_config.decode_context_parallel_size == 1, (
            "DCP not support sliding window."
        )
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
        )
        return max_blocks * self.page_size_bytes
''',
    '''    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # Sliding-window GQA/MQA layers retain TP-local, sequence-replicated
        # caches. Their KV heads differ across TP ranks, so they cannot share
        # the target MLA cache's DCP sequence sharding.
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
        )
        return max_blocks * self.page_size_bytes
''',
)

replace(
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
    '''from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
''',
    '''from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
    )
''',
)

# DFlash verifies a short block of draft tokens with non-causal attention.
# FlashInfer's DCP prefill wrapper already splits cached context from new tokens
# and merges their LSEs; make the new-token ragged attention honor the caller's
# causal flag instead of rejecting this otherwise-supported case.
replace(
    "v1/attention/backends/flashinfer.py",
    '''        self.use_dcp = self.dcp_world_size > 1
        self.dcp_a2a = (
''',
    '''        self.use_dcp = self.dcp_world_size > 1
        if self.kv_cache_spec.sliding_window is not None:
            # DFlash's GQA KV heads are TP-sharded rather than replicated.
            # Keep its short sliding-window sequence cache replicated while
            # target MLA continues to use DCP.
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.use_dcp = False
        self.dcp_a2a = (
''',
)

# The generic speculative-query router used to look only at the target's
# configured DCP size.  The DFlash sliding-window builder above deliberately
# disables sequence DCP because its TP-sharded KV cache is replicated.  Honor
# that effective builder state so its fixed 1+K query can take the SM12x XQA
# decode path (and its stable CUDA-graph buffers) instead of the mutable
# non-causal prefill wrapper.  Target builders still have use_dcp=True and keep
# the conservative one-token threshold.
replace(
    "v1/attention/backend.py",
    '''        if (
            self.vllm_config.parallel_config.decode_context_parallel_size > 1
            and not supports_dcp_with_varlen
        ):
            self.reorder_batch_threshold = 1
''',
    '''        effective_use_dcp = getattr(
            self,
            "use_dcp",
            self.vllm_config.parallel_config.decode_context_parallel_size > 1,
        )
        if effective_use_dcp and not supports_dcp_with_varlen:
            self.reorder_batch_threshold = 1
''',
)

# CUDA-graph capability discovery runs over every target and draft cache
# group before DFlash receives its private graph manager.  The generic
# FlashInfer query used to see the target's configured DCP2 and downgrade the
# whole runner to single-token-only, even for DFlash's sliding-window group.
# That group is explicitly sequence-replicated above (effective DCP1), and
# SM12x XQA supports its fixed 1+K query.  Keep the conservative DCP limit for
# all target/non-sliding attention.
replace(
    "v1/attention/backends/flashinfer.py",
    '''        if is_sm12x and vllm_config.parallel_config.decode_context_parallel_size > 1:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
''',
    '''        replicated_sliding_window = (
            isinstance(kv_cache_spec, AttentionSpec)
            and kv_cache_spec.sliding_window is not None
        )
        if (
            is_sm12x
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and not replicated_sliding_window
        ):
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
''',
)

# DFlash2 selects FLASH_ATTN for its bidirectional sliding-window block on
# this image. Like the FlashInfer path above, its TP-local KV must not inherit
# the target model's sequence DCP coordinates.
replace(
    "v1/attention/backends/flash_attn.py",
    '''        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        # Fused draft decode reuses the captured metadata object across draft
''',
    '''        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        if kv_cache_spec.sliding_window is not None:
            # DFLASH-FA-DCP-ISOLATION: draft heads are TP-sharded and their
            # short sliding-window sequence cache is replicated, not DCP-sharded.
            self.dcp_world_size = 1
            self.dcp_rank = 0

        # Fused draft decode reuses the captured metadata object across draft
''',
)

replace(
    "v1/attention/backends/flash_attn.py",
    '''        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
''',
    '''        self.num_kv_heads = num_kv_heads
        if sliding_window is not None:
            # DFLASH-FA-DCP-ISOLATION: mirror the metadata builder. The draft
            # cache is sequence-replicated even when target MLA uses DCP2.
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.total_cp_world_size = self.pcp_world_size
            self.total_cp_rank = self.pcp_rank
        if alibi_slopes is not None:
''',
)

replace(
    "v1/attention/backends/flashinfer.py",
    '''        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
''',
    '''        self.num_kv_heads = num_kv_heads
        if sliding_window is not None:
            # AttentionImpl.__new__ installs global DCP coordinates. Override
            # them for TP-sharded sliding-window KV heads; their sequence cache
            # is deliberately replicated on every TP rank.
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.total_cp_world_size = self.pcp_world_size
            self.total_cp_rank = self.pcp_rank
        if alibi_slopes is not None:
''',
)

replace(
    "v1/attention/backends/flashinfer.py",
    '''        disable_split_kv: bool,
    ):
        """Plan the prefill operation with given parameters."""
''',
    '''        disable_split_kv: bool,
        causal: bool = True,
    ):
        """Plan the prefill operation with given parameters."""
''',
)

replace(
    "v1/attention/backends/flashinfer.py",
    '''            causal=True,  # This is newtokens run
''',
    '''            # DFlash draft verification is non-causal within the new
            # token block; ordinary target-model prefill remains causal.
            causal=causal,
''',
)

replace(
    "v1/attention/backends/flashinfer.py",
    '''        if not causal and self.use_dcp:
            raise NotImplementedError(
                "FlashInfer non-causal prefill is not supported with DCP yet."
            )
        if not causal and self.use_trtllm_decode_attention:
''',
    '''        # BatchDCPPrefillWrapper supports DFlash's non-causal new-token
        # block; its plan receives the metadata causal flag below.
        if not causal and self.use_trtllm_decode_attention:
''',
)

replace(
    "v1/attention/backends/flashinfer.py",
    '''    def _get_prefill_wrapper(
        self,
        causal: bool = True,
    ) -> BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper:
        if not causal:
            if self.use_dcp:
                raise NotImplementedError(
                    "FlashInfer non-causal prefill is not supported with DCP yet."
                )
''',
    '''    def _get_prefill_wrapper(
        self,
        causal: bool = True,
    ) -> BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper:
        if not causal:
            if self.use_dcp:
                if self._noncausal_prefill_wrapper is None:
                    self._noncausal_prefill_wrapper = BatchDCPPrefillWrapper(
                        workspace_buffer=self._get_workspace_buffer(),
                        dcp_a2a=self.dcp_a2a,
                    )
                return self._noncausal_prefill_wrapper
''',
)

replace(
    "v1/attention/backends/flashinfer.py",
    '''                        prefill_fixed_split_size=self.prefill_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                    )
                else:
''',
    '''                        prefill_fixed_split_size=self.prefill_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                        causal=attn_metadata.causal,
    )
                else:
''',
)

replace(
    "v1/attention/backends/flashinfer.py",
    '''        lse_context = lse_context.transpose(0, 1).contiguous()

        output_query, lse_query = self._new_tokens.run(
            prefill_query,
            key,
            value,
            return_lse=True,
        )
        lse_query = lse_query.transpose(0, 1).contiguous()
''',
    '''        # FlashInfer exposes both partial LSEs in log2 units, whereas
        # merge_attn_states applies natural exp/log. Convert at this final
        # backend-independent merge boundary; cross-rank correction above
        # deliberately remains in log2.
        lse_context.mul_(0.6931471805599453)
        lse_context = lse_context.transpose(0, 1).contiguous()

        output_query, lse_query = self._new_tokens.run(
            prefill_query,
            key,
            value,
            return_lse=True,
        )
        lse_query.mul_(0.6931471805599453)
        lse_query = lse_query.transpose(0, 1).contiguous()
''',
)

replace(
    "v1/attention/backends/flashinfer.py",
    '''                    assert prefill_wrapper._new_tokens._causal
''',
    '''                    assert (
                        prefill_wrapper._new_tokens._causal
                        == attn_metadata.causal
                    )
''',
)

replace(
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
    '''    is_sparse = True
''',
    '''    is_sparse = True
    can_return_lse_for_decode = True
    # SparkInfer's unified SM120 kernel returns log2(sum(exp2(score))).
    lse_base_on_e = False
''',
)

replace(
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
    '''        topk_indices_physical = cast(
            torch.Tensor,
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            ),
        )
''',
    '''        local_topk_lens: torch.Tensor | None = None
        if self.dcp_world_size > 1:
            topk_indices_physical, local_topk_lens = (
                triton_filter_and_convert_dcp_index(
                    attn_metadata.req_id_per_token[:num_actual_toks],
                    attn_metadata.block_table,
                    topk_indices,
                    dcp_size=self.dcp_world_size,
                    dcp_rank=self.dcp_rank,
                    cp_kv_cache_interleave_size=(
                        attn_metadata.cp_kv_cache_interleave_size
                    ),
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                    return_valid_counts=True,
                )
            )
        else:
            topk_indices_physical = cast(
                torch.Tensor,
                triton_convert_req_index_to_global_index(
                    attn_metadata.req_id_per_token[:num_actual_toks],
                    attn_metadata.block_table,
                    topk_indices,
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                ),
            )
''',
)

replace(
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
    '''            out, _ = run_unified_prefill(
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
''',
    '''            empty_rows: torch.Tensor | None = None
            kernel_topk_lens = local_topk_lens
            if local_topk_lens is not None:
                empty_rows = local_topk_lens == 0
                # The kernel requires at least one valid entry. Its result for an
                # empty shard is discarded below and represented as (0, -inf).
                topk_indices_physical[:, 0] = topk_indices_physical[
                    :, 0
                ].masked_fill(empty_rows, 0)
                kernel_topk_lens = local_topk_lens.clamp(min=1)

            out, lse = run_unified_prefill(
                q=q,
                kv_cache=kv_c_and_k_pe_cache.view(torch.uint8),
                topk_indices=topk_indices_physical,
                topk_length=kernel_topk_lens,
                sm_scale=self.scale,
                page_block_size=attn_metadata.block_size,
                output=output,
                scale_format=ScaleFormat.ARBITRARY_FP32,
                fp8_rope=False,
            )
            if not self.need_to_return_lse_for_decode:
                lse = None
            if empty_rows is not None:
                out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
                assert lse is not None
                lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
            return out, lse
''',
)

replace(
    "model_executor/layers/attention/mla_attention.py",
    '''                seq_lens = (
                    attn_metadata.decode.seq_lens
                    if attn_metadata.decode is not None
                    else cast(torch.Tensor, attn_metadata.seq_lens)[  # type: ignore[attr-defined]
                        : attn_metadata.num_decodes
                    ]
                )
                query_start_loc = attn_metadata.query_start_loc[
                    : attn_metadata.num_decodes + 1
                ]
''',
    '''                if self.impl.is_sparse:
                    # Sparse MLA routes prefills through forward_mqa as well as
                    # decodes.  Use the full request metadata here; slicing by
                    # num_decodes produces empty tensors on chunked-prefill-only
                    # steps and breaks DCP's empty-shard LSE mask.
                    seq_lens = cast(torch.Tensor, attn_metadata.seq_lens)  # type: ignore[attr-defined]
                    query_start_loc = attn_metadata.query_start_loc[
                        : attn_metadata.num_reqs + 1  # type: ignore[attr-defined]
                    ]
                else:
                    decode_metadata = getattr(attn_metadata, "decode", None)
                    seq_lens = (
                        decode_metadata.seq_lens
                        if decode_metadata is not None
                        else cast(torch.Tensor, attn_metadata.seq_lens)[  # type: ignore[attr-defined]
                            : attn_metadata.num_decodes
                        ]
                    )
                    query_start_loc = attn_metadata.query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ]
''',
)

replace(
    "v1/core/single_type_kv_cache_manager.py",
    '''        kwargs["max_admission_blocks_per_request"] = (
            kv_cache_spec.max_admission_blocks_per_request(
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
            )
        )
    manager = manager_class(kv_cache_spec, **kwargs)
''',
    '''        admission_cap = kv_cache_spec.max_admission_blocks_per_request(
            max_in_flight_tokens=max_in_flight_tokens,
            max_model_len=max_model_len,
        )
        dcp_world_size = kwargs.get("dcp_world_size", 1)
        if dcp_world_size > 1 and isinstance(kv_cache_spec, AttentionSpec):
            admission_cap = cdiv(admission_cap, dcp_world_size)
        kwargs["max_admission_blocks_per_request"] = admission_cap
    if isinstance(kv_cache_spec, (MambaSpec, SlidingWindowSpec)):
        # Recurrent state and TP-sharded sliding-window KV are replicated, not
        # sequence-sharded. Only target attention managers use DCP geometry.
        kwargs["dcp_world_size"] = 1
    manager = manager_class(kv_cache_spec, **kwargs)
''',
)

replace(
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
    '''        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )
''',
    '''        # DCP gathers each rank's query heads before attention, then the
        # shared reducer scatters them back. Allocate for the gathered head
        # count here rather than this TP rank's local ``self.num_heads``.
        output = q.new_empty(
            (num_actual_toks, q.shape[1], self.kv_lora_rank),
            dtype=q.dtype,
        )
''',
)
