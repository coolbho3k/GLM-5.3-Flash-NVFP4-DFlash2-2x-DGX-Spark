# Add EXL3 routed experts to the known-good calibrated native-FP4 KV runtime.
ARG BASE_IMAGE=glm53-v14:nvfp4-gscale-tooling
FROM ${BASE_IMAGE}
LABEL glm53.miaai-exl3.commit="eb0469fbb2b49fd7c025f594a3339a121e58f7a9" \
      glm53.serving.profile="e2-native-fp4-dcp2"

COPY vendor/miaai-exl3/patch_exl3_ext_aarch64.py \
     vendor/miaai-exl3/patch_exl3_fat_kernel.py /opt/glm53/
COPY vendor/miaai-exl3/exl3_fat_gemm.cu \
     vendor/miaai-exl3/exl3_fat_gemm.cuh /opt/glm53/exl3-fat-kernel/
ARG EXLLAMAV3_COMMIT=c5d9c657966ffeeaa9353f0cc899f18629da4a13
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV FLASHINFER_CUDA_ARCH_LIST=12.1a
ENV MAX_JOBS=4
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
RUN set -eux; \
    mkdir -p /tmp/exllamav3; \
    curl -fsSL "https://github.com/turboderp-org/exllamav3/archive/${EXLLAMAV3_COMMIT}.tar.gz" \
      | tar -xz -C /tmp/exllamav3 --strip-components=1; \
    python3 /opt/glm53/patch_exl3_ext_aarch64.py /tmp/exllamav3/exllamav3/exllamav3_ext; \
    python3 /opt/glm53/patch_exl3_fat_kernel.py /tmp/exllamav3/exllamav3/exllamav3_ext /opt/glm53/exl3-fat-kernel; \
    export CPATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${CPATH:+:$CPATH}"; \
    export CPLUS_INCLUDE_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"; \
    export C_INCLUDE_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"; \
    cd /tmp/exllamav3; \
    pip install --no-deps --no-build-isolation --no-cache-dir .; \
    python3 -c "import torch, exllamav3_ext; assert hasattr(exllamav3_ext, 'exl3_moe'); assert hasattr(exllamav3_ext, 'exl3_fat_gemm'); assert hasattr(exllamav3_ext, 'exl3_fat_gemm_scatter')"; \
    rm -rf /tmp/exllamav3 /root/.cache/pip

COPY vendor/miaai-exl3/exl3.py \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
COPY vendor/miaai-exl3/patch_model_overrides.py \
     overlay-exl3-fp8/patch_exl3_registry.py /opt/glm53/
RUN python3 /opt/glm53/patch_exl3_registry.py && \
    python3 /opt/glm53/patch_model_overrides.py && \
    python3 -m py_compile \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py

COPY probes/test_exl3_runtime_image.py /opt/glm53/
RUN python3 /opt/glm53/test_exl3_runtime_image.py
