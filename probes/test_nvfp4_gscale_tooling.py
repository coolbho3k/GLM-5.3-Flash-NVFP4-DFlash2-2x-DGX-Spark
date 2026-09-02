#!/usr/bin/env python3
"""CPU-only tests for capture artifacts, G fitting, and strict image patches."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import kv_calibration.runtime as runtime
from kv_calibration.fit import fit_layer, load_capture, split_layer
from kv_calibration.numerics import (
    E4M3_POSITIVE,
    evaluate_groups,
    quantize_groups_four_over_six,
    round_e2m1_satfinite,
    round_e4m3_satfinite,
)


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


B12X_PATCH = _module(
    "gscale_b12x_patch", ROOT / "overlay-dflash2/patch_b12x_nvfp4_gscale.py"
)
VLLM_PATCH = _module(
    "gscale_vllm_patch", ROOT / "overlay-dflash2/patch_vllm_nvfp4_gscale.py"
)


class NumericsTest(unittest.TestCase):
    def test_known_format_values_and_ties(self) -> None:
        self.assertEqual(E4M3_POSITIVE[-1], 448.0)
        rounded, codes = round_e4m3_satfinite(np.asarray([0.0, 1.0, 448.0, 1000.0]))
        np.testing.assert_array_equal(rounded, [0.0, 1.0, 448.0, 448.0])
        np.testing.assert_array_equal(codes, [0x00, 0x38, 0x7E, 0x7E])
        # Midpoints choose the code with an even least-significant mantissa bit.
        np.testing.assert_array_equal(
            round_e2m1_satfinite(np.asarray([0.25, 0.75, 1.25, 1.75, 5.0])),
            [0.0, 1.0, 1.0, 2.0, 4.0],
        )

    def test_four_over_six_never_worse_than_either_local_candidate(self) -> None:
        rng = np.random.default_rng(7)
        groups = rng.normal(size=(256, 16)).astype(np.float32)
        chosen = quantize_groups_four_over_six(groups, latent_scale=0.125)
        chosen_sse = np.sum((chosen.reconstruction - groups) ** 2, axis=1)
        # Force either arm by reconstructing the implementation's scale choice.
        amax = np.max(np.abs(groups), axis=1)
        arm_errors = []
        for divisor in (6.0, 4.0):
            encoded, _ = round_e4m3_satfinite(amax / (divisor * 0.125))
            scale = encoded * 0.125
            inverse = np.divide(1.0, scale, out=np.zeros_like(scale), where=scale != 0)
            q = round_e2m1_satfinite(groups * inverse[:, None])
            arm_errors.append(np.sum((q * scale[:, None] - groups) ** 2, axis=1))
        np.testing.assert_allclose(chosen_sse, np.minimum(*arm_errors), rtol=1e-6)

    def test_global_scale_recovers_tiny_groups(self) -> None:
        groups = np.full((32, 16), 1e-4, dtype=np.float32)
        baseline = evaluate_groups(groups, latent_scale=1.0)
        scaled = evaluate_groups(groups, latent_scale=1.0 / 256.0)
        self.assertLess(scaled["mse"], baseline["mse"])
        self.assertLess(scaled["zero_scale_fraction"], baseline["zero_scale_fraction"])


class ArtifactAndFitTest(unittest.TestCase):
    def test_load_split_and_fit(self) -> None:
        rng = np.random.default_rng(11)
        values = rng.normal(scale=1e-4, size=(200, 16)).astype(np.float32)
        metadata = {
            "schema": "glm-nvfp4-mla-capture-v1",
            "layer": "model.layers.3.self_attn",
            "stratum": "nemotron-natural__prefill",
            "seen": len(values),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.npz"
            np.savez(
                path,
                metadata=np.asarray(json.dumps(metadata)),
                values=values,
                priorities=rng.random(len(values)),
            )
            loaded, sources = load_capture([path], per_stratum_limit=128)
            self.assertEqual(len(sources), 1)
            train, heldout, coverage = split_layer(
                loaded["model.layers.3.self_attn"], holdout_fraction=0.2, seed=5
            )
            self.assertEqual(len(train) + len(heldout), 128)
            self.assertIn("nemotron-natural__prefill", coverage)
            result = fit_layer(
                train,
                heldout,
                global_scales=np.asarray([1.0, 64.0, 256.0]),
                objective="mse",
                blend_relative_weight=0.1,
                chunk_rows=32,
                heldout_gate=True,
            )
            self.assertIn(result["global_scale"], (64.0, 256.0))
            self.assertLess(
                result["heldout"]["selected"]["mse"],
                result["heldout"]["baseline"]["mse"],
            )


class PatcherTest(unittest.TestCase):
    def test_b12x_patch_is_complete_and_idempotent(self) -> None:
        # Each strict replacement anchor is intentionally unique in the fixture.
        fixture = "\nUNIQUE\n".join(old for _, old, _ in B12X_PATCH.REPLACEMENTS)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "kv_cache.py"
            target.write_text(fixture)
            self.assertTrue(B12X_PATCH.patch(target))
            result = target.read_text()
            self.assertIn(B12X_PATCH.MARKER, result)
            self.assertIn("latent_scale=latent_scale", result)
            self.assertIn("\n        6,\n", result)
            self.assertFalse(B12X_PATCH.patch(target))
            self.assertEqual(result, target.read_text())

    def test_vllm_patch_is_complete_and_idempotent(self) -> None:
        attention_anchor = """            # MLA Args
            q_lora_rank=self.q_lora_rank,
"""
        adapter_anchors = [
            """from vllm.v1.attention.backends.mla.sparse_utils import (
""",
            """        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        # Skip-topk layers are built with indexer=None and get the shared
""",
            """                scale_format=ScaleFormat.NVFP4_E4M3,
                fp8_rope=False,
            )
""",
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
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attention = root / "model_executor/layers/attention/mla_attention.py"
            adapter = root / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
            attention.parent.mkdir(parents=True)
            adapter.parent.mkdir(parents=True)
            attention.write_text(attention_anchor)
            adapter.write_text("\nUNIQUE\n".join(adapter_anchors))
            self.assertTrue(VLLM_PATCH.patch(root))
            self.assertIn("layer_name=prefix", attention.read_text())
            self.assertIn(VLLM_PATCH.MARKER, adapter.read_text())
            self.assertFalse(VLLM_PATCH.patch(root))


if __name__ == "__main__":
    unittest.main()
