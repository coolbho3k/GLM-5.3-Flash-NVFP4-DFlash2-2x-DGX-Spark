#!/usr/bin/env python3
"""Bound per-layer B12X graph workspaces and keep large prefills eager."""

from pathlib import Path


VLLM_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "fused_moe/experts/flashinfer_b12x_moe.py"
)
FLASHINFER_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/b12x_moe.py"
)


def replace_once(old: str, new: str) -> None:
    global source
    if new in source:
        return
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one B12X workspace patch site, found {count}")
    source = source.replace(old, new, 1)


source = VLLM_TARGET.read_text()
replace_once("from typing import Any\n", "import os\nfrom typing import Any\n")
replace_once(
    """            use_cuda_graph=True,\n            max_num_tokens=self.max_num_tokens,\n""",
    """            # Allocate graph-stable buffers only for the explicitly bounded\n            # decode capture width. Large prefill calls use FlashInfer's eager\n            # cached workspace through the hybrid path patched below.\n            use_cuda_graph=os.getenv("VLLM_B12X_USE_CUDA_GRAPH", "0") == "1",\n            max_num_tokens=(\n                int(\n                    os.getenv(\n                        "VLLM_B12X_CUDA_GRAPH_MAX_TOKENS", str(self.max_num_tokens)\n                    )\n                )\n                if os.getenv("VLLM_B12X_USE_CUDA_GRAPH", "0") == "1"\n                else self.max_num_tokens\n            ),\n""",
)
if "            use_cuda_graph=True,\n" in source:
    raise RuntimeError("unsafe unbounded B12X eager workspace survived patch")
VLLM_TARGET.write_text(source)

source = FLASHINFER_TARGET.read_text()
replace_once("from typing import Any, Optional, Tuple\n", "import os\nfrom typing import Any, Optional, Tuple\n")
replace_once(
    """class B12xMoEWrapper:\n""",
    """# MoE layers execute serially on one serving stream, so identical graph\n# scratch/output buffers can be shared safely across wrappers and captures.\n_B12X_GRAPH_BUFFER_CACHE: dict[tuple, tuple] = {}\n\n\nclass B12xMoEWrapper:\n""",
)
replace_once(
    """        max_routed_rows = self.max_num_tokens * self.top_k\n""",
    """        graph_buffer_key = (\n            str(torch.device(self.device)),\n            self.num_experts,\n            self.num_local_experts,\n            self.top_k,\n            self.hidden_size,\n            self.intermediate_size,\n            self.quant_mode,\n            self.activation,\n            self.output_dtype,\n            self.max_num_tokens,\n        )\n        cached = (
            _B12X_GRAPH_BUFFER_CACHE.get(graph_buffer_key)
            if os.getenv("VLLM_B12X_USE_CUDA_GRAPH", "0") == "1"
            else None
        )\n        if cached is not None:\n            self._static_workspace, self._dynamic_workspace, self._moe_output = cached\n            return\n\n        max_routed_rows = self.max_num_tokens * self.top_k\n""",
)
replace_once(
    """            self._moe_output = torch.empty(\n                (self.max_num_tokens, self.hidden_size),\n                dtype=self.output_dtype,\n                device=self.device,\n            )\n            return\n""",
    """            self._moe_output = torch.empty(\n                (self.max_num_tokens, self.hidden_size),\n                dtype=self.output_dtype,\n                device=self.device,\n            )\n            _B12X_GRAPH_BUFFER_CACHE[graph_buffer_key] = (\n                self._static_workspace, self._dynamic_workspace, self._moe_output\n            )\n            return\n""",
)
replace_once(
    """        self._moe_output = torch.empty(\n            (self.max_num_tokens, self.hidden_size),\n            dtype=self.output_dtype,\n            device=self.device,\n        )\n\n    @flashinfer_api(trace=b12x_moe_wrapper_run_trace)\n""",
    """        self._moe_output = torch.empty(\n            (self.max_num_tokens, self.hidden_size),\n            dtype=self.output_dtype,\n            device=self.device,\n        )\n        _B12X_GRAPH_BUFFER_CACHE[graph_buffer_key] = (\n            self._static_workspace, self._dynamic_workspace, self._moe_output\n        )\n\n    @flashinfer_api(trace=b12x_moe_wrapper_run_trace)\n""",
)
replace_once(
    """        if self.use_cuda_graph and num_tokens > self.max_num_tokens:\n            raise ValueError(\n                f"num_tokens ({num_tokens}) exceeds max_num_tokens "\n                f"({self.max_num_tokens})"\n            )\n\n        if self.use_cuda_graph:\n            moe_output = self._moe_output[:num_tokens]\n        else:\n            if _is_cuda_graph_capturing():\n                raise RuntimeError(\n                    "B12xMoEWrapper must be constructed with use_cuda_graph=True "\n                    "to run during CUDA graph capture."\n                )\n            moe_output = torch.empty(\n                (num_tokens, self.hidden_size),\n                dtype=self.output_dtype,\n                device=x.device,\n            )\n""",
    """        graph_compatible_call = (\n            self.use_cuda_graph and num_tokens <= self.max_num_tokens\n        )\n        if _is_cuda_graph_capturing() and not graph_compatible_call:\n            raise RuntimeError(\n                "B12xMoEWrapper CUDA graph capture requires use_cuda_graph=True "\n                f"and num_tokens <= {self.max_num_tokens}, got {num_tokens}."\n            )\n\n        if graph_compatible_call:\n            moe_output = self._moe_output[:num_tokens]\n        else:\n            # Mixed/prefill batches larger than the decode graph capacity stay\n            # eager and use FlashInfer's shape-keyed shared workspace cache.\n            moe_output = torch.empty(\n                (num_tokens, self.hidden_size),\n                dtype=self.output_dtype,\n                device=x.device,\n            )\n""",
)
replace_once(
    """        workspace = None\n        if self.use_cuda_graph:\n""",
    """        workspace = None\n        if graph_compatible_call:\n""",
)
FLASHINFER_TARGET.write_text(source)
print("Patched bounded vLLM/FlashInfer B12X graph workspaces")
