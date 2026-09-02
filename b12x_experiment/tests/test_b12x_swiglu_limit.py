#!/usr/bin/env python3
"""Regression checks for the GLM clamp plumbing into FlashInfer B12X."""

from __future__ import annotations

import inspect

from flashinfer.fused_moe import B12xMoEWrapper
from flashinfer.fused_moe.cute_dsl import b12x_moe
from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
    FlashInferB12xExperts,
)
from vllm.model_executor.layers.fused_moe.oracle import nvfp4


wrapper_signature = inspect.signature(B12xMoEWrapper.__init__)
assert "swiglu_limit" in wrapper_signature.parameters

adapter_source = inspect.getsource(FlashInferB12xExperts)
assert "self._swiglu_limit = moe_config.swiglu_limit" in adapter_source
assert "swiglu_limit=self._swiglu_limit" in adapter_source
assert "VLLM_B12X_USE_CUDA_GRAPH" in adapter_source
assert "use_cuda_graph=True" not in adapter_source
assert "else self.max_num_tokens" in adapter_source
assert "VLLM_B12X_CUDA_GRAPH_MAX_TOKENS" in adapter_source

wrapper_source = inspect.getsource(B12xMoEWrapper.run)
allocation_source = inspect.getsource(B12xMoEWrapper._allocate_buffers)
assert hasattr(b12x_moe, "_B12X_GRAPH_BUFFER_CACHE")
assert "_B12X_GRAPH_BUFFER_CACHE.get" in allocation_source
assert "_B12X_GRAPH_BUFFER_CACHE[graph_buffer_key]" in allocation_source
assert "graph_compatible_call" in wrapper_source
assert "if graph_compatible_call:" in wrapper_source

oracle_source = inspect.getsource(nvfp4.select_nvfp4_moe_backend)
clamp_set = oracle_source.split("NVFP4_BACKENDS_WITH_CLAMP = {", 1)[1].split("}", 1)[0]
assert "NvFp4MoeBackend.FLASHINFER_B12X" in clamp_set

print("B12X clamp API, adapter plumbing, and capability oracle checks passed")
