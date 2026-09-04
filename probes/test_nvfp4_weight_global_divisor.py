#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_nvfp4_checkpoint import build_shard, packed_plan
from optimize_nvfp4_rounding import (
    dequant_target,
    optimize_groups,
    select_global_divisor,
    source_names,
    source_reference,
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

    def test_bf16_source_is_direct_and_strict(self) -> None:
        weight = torch.tensor(
            [[-0.5, 0.25], [0.125, 0.75]], dtype=torch.bfloat16
        )
        reference = source_reference(weight, None, source_format="bf16")
        self.assertEqual(reference.dtype, torch.float32)
        self.assertTrue(torch.equal(reference, weight.float()))
        with self.assertRaisesRegex(ValueError, "expected torch.bfloat16"):
            source_reference(weight.float(), None, source_format="bf16")

    def test_bf16_plan_does_not_require_scale_shards(self) -> None:
        source_weight, source_scale = source_names(
            3, 0, "down_proj", source_format="bf16"
        )
        self.assertIsNone(source_scale)
        packed, scale, global_scale = target_names(3, 0, "down_proj")
        source_index = {source_weight: "bf16.safetensors"}
        target_index = {
            packed: "target.safetensors",
            scale: "target.safetensors",
            global_scale: "target.safetensors",
        }
        plan = packed_plan(
            source_index,
            target_index,
            include_global_divisor=False,
            source_format="bf16",
        )
        entry = plan["target.safetensors"][0]
        self.assertEqual(entry["source_weight_shard"], "bf16.safetensors")
        self.assertIsNone(entry["source_scale_name"])
        self.assertIsNone(entry["source_scale_shard"])

    def test_bf16_build_shard_end_to_end_on_cpu(self) -> None:
        source_weight, _ = source_names(
            3, 0, "down_proj", source_format="bf16"
        )
        packed, scale, global_scale = target_names(3, 0, "down_proj")
        source_shard = "source.safetensors"
        target_shard = "target.safetensors"
        source_index = {source_weight: source_shard}
        target_index = {
            packed: target_shard,
            scale: target_shard,
            global_scale: target_shard,
        }
        plan = packed_plan(
            source_index,
            target_index,
            include_global_divisor=False,
            source_format="bf16",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            target_root = root / "target"
            output_root = root / "output"
            source_root.mkdir()
            target_root.mkdir()
            output_root.mkdir()
            weight = torch.linspace(-0.5, 0.5, 32).reshape(2, 16).bfloat16()
            save_file({source_weight: weight}, source_root / source_shard)
            save_file(
                {
                    packed: torch.zeros((2, 8), dtype=torch.uint8),
                    scale: torch.ones((2, 1)).to(torch.float8_e4m3fn),
                    global_scale: torch.tensor(1.0),
                },
                target_root / target_shard,
                metadata={"format": "pt"},
            )
            report = io.StringIO()
            result = build_shard(
                source_root=source_root,
                source_index=source_index,
                source_format="bf16",
                target_root=target_root,
                target_index=target_index,
                output_root=output_root,
                target_shard=target_shard,
                entries=plan[target_shard],
                device=torch.device("cpu"),
                scale_radius_below=16,
                scale_radius_above=8,
                row_chunk=2,
                global_divisor_steps_per_octave=0,
                global_divisor_search_rows=2,
                global_divisor_heldout_tolerance=0.0,
                report_handle=report,
            )
            self.assertEqual(result["matrices"], 1)
            report_row = json.loads(report.getvalue())
            self.assertEqual(report_row["source_format"], "bf16")
            with safe_open(
                output_root / target_shard, framework="pt", device="cpu"
            ) as handle:
                self.assertEqual(handle.get_tensor(packed).dtype, torch.uint8)
                self.assertEqual(
                    handle.get_tensor(scale).dtype, torch.float8_e4m3fn
                )


if __name__ == "__main__":
    unittest.main()
