"""Enable GLM's compact MXFP4 indexer cache on GB10 / SM121."""

from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")


def replace(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"FP4 indexer patch anchor missing in {target}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1))


replace(
    "models/glm5next/nvidia/attention.py",
    "from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8\n",
    '''from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8
from b12x.attention.dsa_indexer.sm121_mxfp4 import fwht128_quant_mxfp4
''',
)

replace(
    "models/glm5next/nvidia/attention.py",
    '''    return (weights.unsqueeze(-1) * q_scale * scale).squeeze(-1)


@torch.compile(**_INDEXER_COMPILE)
def _pad_indexer_heads''',
    '''    return (weights.unsqueeze(-1) * q_scale * scale).squeeze(-1)


@torch.compile(**_INDEXER_COMPILE)
def _fused_indexer_weight_scale_fp4(
    weights: torch.Tensor, scale: float
) -> torch.Tensor:
    # MXFP4 has four per-block Q scales which must stay in the dot product;
    # weights therefore carry only the model's scalar attention/head factor.
    return weights * scale


@torch.compile(**_INDEXER_COMPILE)
def _pad_indexer_heads''',
)

replace(
    "models/glm5next/nvidia/attention.py",
    '''        self.index_kpool = config.index_kpool
        self.q_lora_rank = q_lora_rank  # 1536
''',
    '''        self.index_kpool = config.index_kpool
        self.q_lora_rank = q_lora_rank  # 1536
        self.use_fp4_cache = (
            vllm_config.attention_config.use_fp4_indexer_cache
        )
''',
)

replace(
    "models/glm5next/nvidia/attention.py",
    '''        # NOTE: (zyongye) we use fp8 naive cache,
        #       where we store value in fp8 and scale in fp32
        #       per self.quant_block_size element
        self.k_cache = Glm5NextIndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
''',
    '''        # FP8 is 128 value bytes + one FP32 scale. MXFP4 is 64 packed
        # value bytes + four UE8M0 scales (one per 32 values).
        index_cache_width = (
            self.head_dim // 2 + self.head_dim // 32
            if self.use_fp4_cache
            else self.head_dim + self.head_dim // self.quant_block_size * 4
        )
        self.k_cache = Glm5NextIndexerCache(
            head_dim=index_cache_width,
''',
)

replace(
    "models/glm5next/nvidia/attention.py",
    '''            self.max_total_seq_len,
            self.topk_indices_buffer,
            tail_cache=self.tail_cache,
''',
    '''            self.max_total_seq_len,
            self.topk_indices_buffer,
            use_fp4_cache=self.use_fp4_cache,
            tail_cache=self.tail_cache,
''',
)

replace(
    "models/glm5next/nvidia/attention.py",
    '''        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = fwht128_quant_fp8(q)
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights = _fused_indexer_weight_scale(
            weights, q_scale, self.softmax_scale * self.n_head**-0.5
        )
''',
    '''        q = q.view(-1, self.head_dim)
        index_scale = self.softmax_scale * self.n_head**-0.5
        if self.use_fp4_cache:
            q_packed, q_scale_bytes = fwht128_quant_mxfp4(q)
            q_packed = q_packed.view(-1, self.n_head, self.head_dim // 2)
            # Four contiguous UE8M0 bytes are passed through the custom-op
            # schema as one int32, matching vLLM's existing FP4 convention.
            q_scale_packed = q_scale_bytes.view(torch.int32).view(
                -1, self.n_head
            )
            weights = _fused_indexer_weight_scale_fp4(weights, index_scale)
            q_quant = (q_packed, q_scale_packed)
        else:
            q_fp8, q_scale = fwht128_quant_fp8(q)
            q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
            q_scale = q_scale.view(-1, self.n_head, 1)
            weights = _fused_indexer_weight_scale(weights, q_scale, index_scale)
            q_quant = q_fp8
''',
)

replace(
    "models/glm5next/nvidia/attention.py",
    '''        if self.n_head < 32:
            pad = 32 - self.n_head
            q_fp8 = _pad_indexer_heads(q_fp8, pad)
            weights = _pad_indexer_heads(weights, pad)

        return self.indexer_op(
            hidden_states,
            q_fp8,
''',
    '''        if self.n_head < 32:
            pad = 32 - self.n_head
            if self.use_fp4_cache:
                q_values, q_scales = q_quant
                q_quant = (
                    _pad_indexer_heads(q_values, pad),
                    _pad_indexer_heads(q_scales, pad),
                )
            else:
                q_quant = _pad_indexer_heads(q_quant, pad)
            weights = _pad_indexer_heads(weights, pad)

        return self.indexer_op(
            hidden_states,
            q_quant,
''',
)

replace(
    "v1/attention/backends/mla/indexer.py",
    '''        assert (
            current_platform.is_device_capability_family(100)
            or not self.use_fp4_indexer_cache
        ), (
            "use_fp4_indexer_cache requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )
''',
    '''        assert (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability_family(120)
            or not self.use_fp4_indexer_cache
        ), (
            "use_fp4_indexer_cache requires Blackwell (SM100 or SM12x)."
        )
''',
)

replace(
    "v1/attention/backends/mla/indexer.py",
    '''        self.use_flattening = not current_platform.is_device_capability_family(
            100
        ) and next_n not in (1, 2)
''',
    '''        # DFLASH-SM121-NATIVE-SPEC: the custom MXFP4 paged reader accepts
        # arbitrary next_n, unlike the generic DeepGEMM SM12x implementation.
        sm121_native_fp4 = (
            current_platform.is_device_capability_family(120)
            and self.use_fp4_indexer_cache
        )
        self.use_flattening = not (
            current_platform.is_device_capability_family(100)
            or sm121_native_fp4
        ) and next_n not in (1, 2)
''',
)

replace(
    "v1/attention/backends/mla/indexer.py",
    '''            if current_platform.is_cuda() and has_deep_gemm():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
''',
    '''            if (
                current_platform.is_cuda()
                and has_deep_gemm()
                and not (
                    current_platform.is_device_capability_family(120)
                    and self.use_fp4_indexer_cache
                )
            ):
                # SM121 MXFP4 uses its Triton paged reader and does not consume
                # DeepGEMM scheduler metadata (which is limited to next_n <= 2).
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
''',
)
