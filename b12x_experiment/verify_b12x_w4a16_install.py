#!/usr/bin/env python3
"""Fail-closed source/API checks for the GLM B12X W4A16 experiment image."""

from __future__ import annotations

import inspect
import json

from flashinfer.fused_moe import B12xMoEWrapper
from flashinfer.fused_moe.cute_dsl import b12x_moe
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
    FlashInferB12xExperts,
)
from vllm.model_executor.layers.fused_moe.oracle import nvfp4


wrapper_init = inspect.signature(B12xMoEWrapper.__init__)
required_parameters = {
    "expert_map",
    "num_local_experts",
    "quant_mode",
    "swiglu_limit",
}
missing = sorted(required_parameters - set(wrapper_init.parameters))
assert not missing, f"B12X wrapper is missing required parameters: {missing}"

adapter_source = inspect.getsource(FlashInferB12xExperts)
for needle in (
    "self.use_ep = moe_config.moe_parallel_config.use_ep",
    "return not moe_parallel_config.enable_eplb",
    "def supports_expert_map(self) -> bool:",
    "return True",
    "expert_map=expert_map if self.use_ep else None",
    'quant_mode="w4a16" if self.use_ep else "nvfp4"',
    "self._swiglu_limit = moe_config.swiglu_limit",
    "swiglu_limit=self._swiglu_limit",
    "VLLM_B12X_USE_CUDA_GRAPH",
    "VLLM_B12X_CUDA_GRAPH_MAX_TOKENS",
):
    assert needle in adapter_source, f"missing adapter guard: {needle}"
assert "use_cuda_graph=True" not in adapter_source

wrapper_source = inspect.getsource(b12x_moe)
for needle in (
    "def _prepare_ep_expert_map(",
    "expert_map=self.expert_map",
    "_B12X_GRAPH_BUFFER_CACHE",
    "graph_compatible_call",
):
    assert needle in wrapper_source, f"missing wrapper guard: {needle}"

dispatch_source = inspect.getsource(moe_dispatch)
assert "expert_map: torch.Tensor | None = None" in dispatch_source
assert "expert_map=expert_map" in dispatch_source
assert "quant_mode='w4a16'" in dispatch_source

oracle_source = inspect.getsource(nvfp4.select_nvfp4_moe_backend)
clamp_set = oracle_source.split("NVFP4_BACKENDS_WITH_CLAMP = {", 1)[1].split(
    "}", 1
)[0]
assert "NvFp4MoeBackend.FLASHINFER_B12X" in clamp_set

print(
    json.dumps(
        {
            "ok": True,
            "expert_map": True,
            "quant_mode": "w4a16",
            "swiglu_limit": True,
            "bounded_workspace": True,
        },
        sort_keys=True,
    )
)
