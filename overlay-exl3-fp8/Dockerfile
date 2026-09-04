# Reproducible EXL3 routed experts + native 528-byte FP8 MLA + TP2/DCP2.
ARG BASE_IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
FROM ${BASE_IMAGE}
LABEL glm53.miaai-exl3.commit="eb0469fbb2b49fd7c025f594a3339a121e58f7a9" \
      glm53.serving.profile="e2-fp8-dcp2"

# Compile the pinned EXL3 extension before the frequently edited cache
# overlays so a Python/kernel iteration does not trigger a five-minute rebuild.
COPY vendor/miaai-exl3/patch_exl3_ext_aarch64.py vendor/miaai-exl3/patch_exl3_fat_kernel.py vendor/miaai-exl3/patch_exl3_decode_pipeline.py /opt/glm53/
COPY vendor/miaai-exl3/exl3_fat_gemm.cu vendor/miaai-exl3/exl3_fat_gemm.cuh /opt/glm53/exl3-fat-kernel/
ARG EXLLAMAV3_COMMIT=c5d9c657966ffeeaa9353f0cc899f18629da4a13
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV FLASHINFER_CUDA_ARCH_LIST=12.1a
ENV MAX_JOBS=8
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
RUN set -eux; mkdir -p /tmp/exllamav3; \
    curl -fsSL "https://github.com/turboderp-org/exllamav3/archive/${EXLLAMAV3_COMMIT}.tar.gz" | tar -xz -C /tmp/exllamav3 --strip-components=1; \
    python3 /opt/glm53/patch_exl3_ext_aarch64.py /tmp/exllamav3/exllamav3/exllamav3_ext; \
    python3 /opt/glm53/patch_exl3_fat_kernel.py /tmp/exllamav3/exllamav3/exllamav3_ext /opt/glm53/exl3-fat-kernel; \
    python3 /opt/glm53/patch_exl3_decode_pipeline.py /tmp/exllamav3/exllamav3/exllamav3_ext; \
    export CPATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${CPATH:+:$CPATH}"; \
    export CPLUS_INCLUDE_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"; \
    export C_INCLUDE_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"; \
    cd /tmp/exllamav3; pip install --no-deps --no-build-isolation --no-cache-dir .; \
    python3 -c "import torch; import exllamav3_ext as e; assert hasattr(e, 'exl3_moe'); assert hasattr(e, 'exl3_moe_max_concurrency'); assert hasattr(e, 'exl3_fat_gemm'); assert hasattr(e, 'exl3_fat_gemm_scatter'); assert e.glm53_fast_moe_version() == 1"; \
    rm -rf /tmp/exllamav3 /root/.cache/pip

# SparkInfer is pinned and patched by build-exl3-fp8-dcp2-image.sh.
COPY --from=sparkinfer b12x /usr/local/lib/python3.12/dist-packages/b12x
COPY overlay-exl3-fp8/b12x_mxfp8.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/b12x.py
COPY overlay-exl3-fp8/fp8_zero_rope.py /usr/local/lib/python3.12/dist-packages/b12x/attention/_shared/mla/fp8_zero_rope.py
COPY overlay-dflash2/qwen3_dflash2.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py
COPY overlay-dflash2/dflash2/speculator.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py
COPY overlay-dflash2/sm121_mxfp4_indexer.py /usr/local/lib/python3.12/dist-packages/b12x/attention/dsa_indexer/sm121_mxfp4.py
COPY docker/sparse_attn_indexer_kpool_sm121.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py
COPY overlay-dflash2/compact_kda_replay.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/ops/compact_kda_replay.py

COPY overlay-exl3-fp8/patch_b12x_fp8_topk_2176.py overlay-dflash2/patch_kv_accounting.py overlay-dflash2/patch_vllm_glm_ct_fp8_passthrough.py overlay-exl3-fp8/patch_vllm_fp8_zero_rope.py overlay-dflash2/patch_vllm_sm121_fp4_indexer.py overlay-exl3-fp8/patch_vllm_fp8_dcp.py overlay-dflash2/patch_vllm_kda_fp16_state.py overlay-dflash2/patch_vllm_compact_kda_replay.py overlay-dflash2/patch_compact_replay_production.py overlay-dflash2/patch_resident_dtype_inventory.py overlay-dflash2/patch_long_context_replicated_groups.py overlay-dflash2/patch_long_context_indexer_workspace.py /tmp/
RUN python3 /tmp/patch_b12x_fp8_topk_2176.py && python3 /tmp/patch_kv_accounting.py && python3 /tmp/patch_vllm_glm_ct_fp8_passthrough.py && python3 /tmp/patch_vllm_fp8_zero_rope.py && python3 /tmp/patch_vllm_sm121_fp4_indexer.py && python3 /tmp/patch_vllm_fp8_dcp.py && python3 /tmp/patch_vllm_kda_fp16_state.py && python3 /tmp/patch_vllm_compact_kda_replay.py && python3 /tmp/patch_compact_replay_production.py && python3 /tmp/patch_resident_dtype_inventory.py && python3 /tmp/patch_long_context_replicated_groups.py && python3 /tmp/patch_long_context_indexer_workspace.py && rm /tmp/patch_*.py && python3 -m compileall -q /usr/local/lib/python3.12/dist-packages/b12x /usr/local/lib/python3.12/dist-packages/vllm

COPY overlay-exl3-fp8/patch_vllm_b12x_mxfp8.py overlay-exl3-fp8/patch_dflash_mxfp8_context.py overlay-exl3-fp8/patch_dflash_only_cudagraph_lifecycle.py /tmp/
RUN python3 /tmp/patch_vllm_b12x_mxfp8.py && python3 /tmp/patch_dflash_mxfp8_context.py && python3 /tmp/patch_dflash_only_cudagraph_lifecycle.py && rm /tmp/patch_vllm_b12x_mxfp8.py /tmp/patch_dflash_mxfp8_context.py /tmp/patch_dflash_only_cudagraph_lifecycle.py && python3 -m compileall -q /usr/local/lib/python3.12/dist-packages/vllm

COPY vendor/miaai-exl3/exl3.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
COPY vendor/miaai-exl3/patch_model_overrides.py overlay-exl3-fp8/patch_exl3_registry.py overlay-exl3-fp8/patch_spinwait.py /opt/glm53/
RUN python3 /opt/glm53/patch_exl3_registry.py && python3 /opt/glm53/patch_model_overrides.py && GLM53_SPINWAIT_MS=stock python3 /opt/glm53/patch_spinwait.py --preflight && python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py

COPY probes/test_exl3_fp8_dcp2_image.py /opt/glm53/
COPY probes/test_exl3_e2_kernel.py /opt/glm53/
COPY probes/test_fp8_zero_rope_writer.py /opt/glm53/
COPY probes/test_fp8_zero_rope_b12x.py /opt/glm53/
COPY probes/test_spinwait_patch.py /opt/glm53/
COPY probes/test_mxfp8_drafter_image.py /opt/glm53/
RUN python3 /opt/glm53/test_exl3_fp8_dcp2_image.py && python3 /opt/glm53/test_spinwait_patch.py && python3 /opt/glm53/test_mxfp8_drafter_image.py

COPY --chmod=755 overlay-exl3-fp8/entrypoint.sh /opt/glm53/entrypoint.sh
ENTRYPOINT ["/opt/glm53/entrypoint.sh"]
