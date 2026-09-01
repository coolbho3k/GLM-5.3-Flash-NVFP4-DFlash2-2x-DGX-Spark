#!/usr/bin/env python3
"""Static image gate for EXL3 + zero-RoPE FP8 + DCP2 composition."""

from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages")
VLLM = SITE / "vllm"


def main() -> None:
    from b12x.attention._shared.mla.traits import (
        ComputeMode,
        ModelType,
        ScaleFormat,
        make_unified_traits,
    )
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config

    assert "exl3" in QUANTIZATION_METHODS
    assert get_quantization_config("exl3") is Exl3Config
    traits = make_unified_traits(
        ModelType.GLM_ZERO_ROPE,
        ComputeMode.FP8,
        ScaleFormat.ARBITRARY_FP32,
        fp8_rope=False,
    )
    assert (traits.kv_gmem_stride, traits.kv_smem_stride, traits.d_rope) == (
        528,
        528,
        0,
    )
    import exllamav3_ext

    assert hasattr(exllamav3_ext, "exl3_moe")
    assert hasattr(exllamav3_ext, "exl3_moe_max_concurrency")
    kv_interface = (VLLM / "v1/kv_cache_interface.py").read_text()
    sparse = (
        VLLM / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
    ).read_text()
    flash_attn = (VLLM / "v1/attention/backends/flash_attn.py").read_text()
    indexer = (VLLM / "v1/attention/backends/mla/indexer.py").read_text()
    manager = (VLLM / "v1/core/single_type_kv_cache_manager.py").read_text()
    kv_cache_utils = (VLLM / "v1/core/kv_cache_utils.py").read_text()
    runner = (VLLM / "v1/worker/gpu_model_runner.py").read_text()
    v2_runner = (VLLM / "v1/worker/gpu/model_runner.py").read_text()
    runner += v2_runner
    assert "return self.block_size * 528" in kv_interface
    assert "concat_and_cache_fp8_mla_zero_rope" in sparse
    assert "ScaleFormat.ARBITRARY_FP32" in sparse
    assert "triton_filter_and_convert_dcp_index" in sparse
    assert flash_attn.count("DFLASH-FA-DCP-ISOLATION") == 2
    prefill = (
        SITE / "b12x/attention/_shared/mla/prefill.py"
    ).read_text()
    assert "if _mg_glm and topk in (512, 1024, 2048, 2176):" in prefill
    assert "get_compressed_pool_slot_mapping_dcp" in indexer
    assert "current_platform.is_device_capability_family(120)" in indexer
    assert "DFLASH-SM121-NATIVE-SPEC" in indexer
    assert "sm121_native_fp4" in indexer
    glm_attention = (
        VLLM / "models/glm5next/nvidia/attention.py"
    ).read_text()
    glm_kda = (VLLM / "models/glm5next/nvidia/kda.py").read_text()
    mamba_abstract = (
        VLLM / "model_executor/layers/mamba/abstract.py"
    ).read_text()
    assert "fwht128_quant_mxfp4" in glm_attention
    assert "self.use_fp4_cache" in glm_attention
    assert "VLLM_COMPACT_SPEC_REPLAY" in glm_kda
    assert "_compact_replay_accepted" in glm_kda
    assert "fixed maximum shapes" in glm_kda
    assert "VLLM_COMPACT_SPEC_REPLAY" in mamba_abstract
    kpool = (
        VLLM / "model_executor/layers/sparse_attn_indexer_kpool.py"
    ).read_text()
    assert "SM121 FP4 indexer insertion requires GLM kpool compression" in kpool
    assert "kpool_compress_and_write_cache_mxfp4" in kpool
    assert "DFLASH-SM121-NATIVE-SPEC" in kpool
    assert "paged_mqa_logits_mxfp4" in kpool
    assert "Unfused FP4 Insert is not supported yet" not in kpool
    assert "admission_cap = cdiv(admission_cap, dcp_world_size)" in manager
    assert "DFlash2 drafter KV: padded slot-share block=%d" in kv_cache_utils
    assert "s.block_size != 64 or s.page_size_padded != mla_page" in kv_cache_utils
    assert "LONG-CONTEXT-REPLICATED-GROUPS" in runner
    assert (
        "if active_replays and input_batch.num_draft_tokens > 0:" in v2_runner
    )
    assert "sequence_mask[: input_batch.num_reqs]" in v2_runner
    assert "GLM-KPOOL-WORKSPACE-BOUND" in indexer
    writer = SITE / "b12x/attention/_shared/mla/fp8_zero_rope.py"
    compile(writer.read_text(), str(writer), "exec")
    print("EXL3 + zero-RoPE FP8 + DCP2 image composition OK")


if __name__ == "__main__":
    main()
