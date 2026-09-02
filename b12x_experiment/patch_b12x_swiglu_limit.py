#!/usr/bin/env python3
"""Wire GLM's SwiGLU clamp into FlashInfer's SM12x fused NVFP4 MoE.

The pinned FlashInfer B12xMoEWrapper and its W4A16 epilogue support a
``swiglu_limit`` for the ordinary ``silu`` activation. The corresponding
vLLM adapter predates that API plumbing and its NVFP4 oracle consequently
rejects B12X whenever a model (including GLM-5.3) requests a clamp.
"""

from pathlib import Path


SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
EXPERTS = SITE / "model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"
ORACLE = SITE / "model_executor/layers/fused_moe/oracle/nvfp4.py"


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if new in source:
        return
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one patch site in {path}, found {count}")
    path.write_text(source.replace(old, new, 1))


replace_once(
    EXPERTS,
    """        self._activation_str = self._ACTIVATION_MAP[activation]\n\n        # Lazily created on first apply() call.\n""",
    """        self._activation_str = self._ACTIVATION_MAP[activation]\n        # FlashInfer SM12x implements gate=min(gate, limit) and clamps up to\n        # [-limit, limit], matching vLLM's swiglu_limit_func.\n        self._swiglu_limit = moe_config.swiglu_limit\n\n        # Lazily created on first apply() call.\n""",
)

replace_once(
    EXPERTS,
    """            num_local_experts=self.num_local_experts,\n            activation=self._activation_str,\n        )\n""",
    """            num_local_experts=self.num_local_experts,\n            activation=self._activation_str,\n            swiglu_limit=self._swiglu_limit,\n        )\n""",
)

replace_once(
    ORACLE,
    """    NVFP4_BACKENDS_WITH_CLAMP = {\n        NvFp4MoeBackend.FLASHINFER_TRTLLM,\n""",
    """    NVFP4_BACKENDS_WITH_CLAMP = {\n        # Installed FlashInfer's SM12x W4A16 epilogue supports this clamp.\n        NvFp4MoeBackend.FLASHINFER_B12X,\n        NvFp4MoeBackend.FLASHINFER_TRTLLM,\n""",
)

print("Patched vLLM FlashInfer B12X SwiGLU clamp integration")
