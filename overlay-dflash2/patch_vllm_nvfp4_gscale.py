#!/usr/bin/env python3
"""Wire capture and calibrated outer scales into vLLM's SM120 MLA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "NVFP4_GSCALE_TOOLING"


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {description} anchor, found {count}")
    return text.replace(old, new, 1)


def patch(root: Path) -> bool:
    attention_path = root / "model_executor/layers/attention/mla_attention.py"
    adapter_path = root / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
    attention = attention_path.read_text()
    adapter = adapter_path.read_text()
    if MARKER in adapter:
        print(f"already patched: {adapter_path}")
        return False

    attention = replace_once(
        attention,
        """            # MLA Args
            q_lora_rank=self.q_lora_rank,
""",
        """            # MLA Args
            layer_name=prefix,
            q_lora_rank=self.q_lora_rank,
""",
        "MLA layer-name forwarding",
    )
    adapter = replace_once(
        adapter,
        """from vllm.v1.attention.backends.mla.sparse_utils import (
""",
        """from vllm.nvfp4_mla_calibration import runtime_for_layer  # NVFP4_GSCALE_TOOLING
from vllm.v1.attention.backends.mla.sparse_utils import (
""",
        "runtime import",
    )
    adapter = replace_once(
        adapter,
        """        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        # Skip-topk layers are built with indexer=None and get the shared
""",
        """        self.kv_scale_format = _kv_scale_format_for_model(model_type)
        if self.qk_rope_head_dim == 0:
            self._nvfp4_runtime = runtime_for_layer(mla_args.get("layer_name", ""))
        else:
            self._nvfp4_runtime = None

        # Skip-topk layers are built with indexer=None and get the shared
""",
        "runtime initialization",
    )
    adapter = replace_once(
        adapter,
        """                scale_format=ScaleFormat.NVFP4_E4M3,
                fp8_rope=False,
            )
""",
        """                scale_format=ScaleFormat.NVFP4_E4M3,
                fp8_rope=False,
                latent_scale=self._nvfp4_runtime.latent_scale,
            )
""",
        "reader outer scale",
    )
    adapter = replace_once(
        adapter,
        """        if self.qk_rope_head_dim == 0:
            from b12x.attention._shared.mla.kv_cache import (
                concat_and_cache_nvfp4_mla_zero_rope,
            )
            concat_and_cache_nvfp4_mla_zero_rope(
                kv_c_normed,
                kv_cache.view(torch.uint8),
                slot_mapping.flatten(),
                scale=k_scale,
            )
""",
        """        if self.qk_rope_head_dim == 0:
            from b12x.attention._shared.mla.kv_cache import (
                concat_and_cache_nvfp4_mla_zero_rope,
            )
            assert self._nvfp4_runtime is not None
            if self._nvfp4_runtime.collector is not None:
                self._nvfp4_runtime.collector.observe(kv_c_normed, slot_mapping)
            concat_and_cache_nvfp4_mla_zero_rope(
                kv_c_normed,
                kv_cache.view(torch.uint8),
                slot_mapping.flatten(),
                scale=k_scale,
                latent_scale=self._nvfp4_runtime.latent_scale,
            )
""",
        "writer capture and outer scale",
    )
    attention_path.write_text(attention)
    adapter_path.write_text(adapter)
    print(f"patched: {attention_path}")
    print(f"patched: {adapter_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    patch(args.root)


if __name__ == "__main__":
    main()
