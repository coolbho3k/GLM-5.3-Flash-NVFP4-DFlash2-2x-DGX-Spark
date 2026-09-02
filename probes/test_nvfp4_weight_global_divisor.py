#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_nvfp4_checkpoint import packed_plan
from optimize_nvfp4_rounding import (
    dequant_target,
    optimize_groups,
    select_global_divisor,
    source_names,
    target_names,
)


class GlobalDivisorSearchTest(unittest.TestCase):
    def test_disabled_search_preserves_divisor_exactly(self) -> None:
        weight = torch.ones((2, 16))
        divisor = torch.tensor(16384.0)
        selected, report = select_global_divisor(
            weight,
            divisor,
            steps_per_octave=0,
            search_rows=2,
            heldout_tolerance=0.0,
            scale_radius_below=16,
            scale_radius_above=8,
            row_chunk=2,
        )
        self.assertTrue(torch.equal(selected, divisor))
        self.assertFalse(report["enabled"])

    def test_search_is_deterministic_and_heldout_gated(self) -> None:
        generator = torch.Generator().manual_seed(7)
        base = torch.randn((1, 256), generator=generator) * 0.035
        weight = base.repeat(32, 1)
        divisor = torch.tensor(16384.0)
        kwargs = {
            "steps_per_octave": 16,
            "search_rows": 32,
            "heldout_tolerance": 0.0,
            "scale_radius_below": 16,
            "scale_radius_above": 8,
            "row_chunk": 16,
        }
        selected, report = select_global_divisor(weight, divisor, **kwargs)
        repeated, repeated_report = select_global_divisor(weight, divisor, **kwargs)
        self.assertTrue(torch.equal(selected, repeated))
        self.assertEqual(
            report["selected_multiplier"],
            repeated_report["selected_multiplier"],
        )
        self.assertEqual(len(report["train_curve"]), 16)
        self.assertLessEqual(
            report["selected_heldout"]["mse"],
            report["original_heldout"]["mse"] + 1e-15,
        )

        packed, scale, _ = optimize_groups(
            weight,
            selected,
            scale_radius_below=16,
            scale_radius_above=8,
            row_chunk=16,
            input_second_moment=None,
        )
        reconstructed = dequant_target(packed, scale, selected)
        self.assertTrue(torch.isfinite(reconstructed).all())

    def test_build_plan_adds_global_destination_only_when_enabled(self) -> None:
        source_weight, source_scale = source_names(3, 0, "down_proj")
        packed, scale, global_scale = target_names(3, 0, "down_proj")
        source_index = {
            source_weight: "source-weight.safetensors",
            source_scale: "source-scale.safetensors",
        }
        target_index = {
            packed: "packed.safetensors",
            scale: "scale.safetensors",
            global_scale: "global.safetensors",
        }
        old_plan = packed_plan(
            source_index, target_index, include_global_divisor=False
        )
        new_plan = packed_plan(
            source_index, target_index, include_global_divisor=True
        )
        self.assertEqual(set(old_plan), {"packed.safetensors", "scale.safetensors"})
        self.assertEqual(
            set(new_plan),
            {"packed.safetensors", "scale.safetensors", "global.safetensors"},
        )
        self.assertTrue(new_plan["global.safetensors"][0]["replace_global"])


if __name__ == "__main__":
    unittest.main()
