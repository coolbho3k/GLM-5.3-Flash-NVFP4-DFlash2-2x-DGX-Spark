#!/usr/bin/env python3

import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from probes.verify_nvfp4_checkpoint import safetensors_payload_size, verify_reports


class VerifyBuildReportsTest(unittest.TestCase):
    def test_payload_size_excludes_variable_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            header = b"{\"tensor\":{}}    "
            payload = b"payload-bytes"
            path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            self.assertEqual(safetensors_payload_size(path), len(payload))

    @staticmethod
    def row(
        expert: int,
        *,
        values: int,
        baseline_mse: float,
        candidate_mse: float,
        multiplier: float,
        heldout_fallback: bool = False,
        full_fallback: bool = False,
    ) -> dict:
        return {
            "layer": 3,
            "expert": expert,
            "projection": "down_proj" if expert == 0 else "up_proj",
            "shape": [values, 1],
            "groups": values,
            "replace_packed": True,
            "values": values,
            "original_global_divisor": 10.0,
            "selected_global_divisor": 10.0 * multiplier,
            "global_divisor_multiplier": multiplier,
            "global_divisor_search": {
                "fell_back": heldout_fallback,
                "selected_multiplier": multiplier,
                "full_matrix_gate": {
                    "baseline_mse": baseline_mse,
                    "selected_mse_before_gate": candidate_mse,
                    "fell_back": full_fallback,
                },
            },
            "search": {
                "lower_boundary_fraction": 0.0,
                "upper_boundary_fraction": 0.0,
            },
        }

    def test_aggregates_unique_packed_rows_and_applied_fallbacks(self) -> None:
        first = self.row(
            0,
            values=100,
            baseline_mse=4.0,
            candidate_mse=1.0,
            multiplier=2.0,
        )
        second = self.row(
            1,
            values=300,
            baseline_mse=1.0,
            candidate_mse=2.0,
            multiplier=1.0,
            heldout_fallback=True,
            full_fallback=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.jsonl"
            report.write_text(
                "\n".join(json.dumps(row) for row in (first, first, second))
                + "\n"
            )
            result = verify_reports(
                [report], {(3, 0, "down_proj"), (3, 1, "up_proj")}
            )

        self.assertEqual(result["report_rows"], 3)
        self.assertEqual(result["duplicate_matrix_report_rows"], 1)
        self.assertEqual(result["duplicate_packed_report_rows"], 1)
        aggregate = result["global_divisor_optimization"]
        self.assertEqual(aggregate["matrices"], 2)
        self.assertEqual(aggregate["changed_divisors"], 1)
        self.assertEqual(aggregate["heldout_gate_fallbacks"], 1)
        self.assertEqual(aggregate["full_matrix_gate_fallbacks"], 1)
        self.assertAlmostEqual(aggregate["baseline_weighted_mse"], 1.75)
        self.assertAlmostEqual(aggregate["selected_weighted_mse"], 1.0)
        self.assertAlmostEqual(
            aggregate["weighted_mse_relative_reduction"], 3.0 / 7.0
        )
        self.assertAlmostEqual(
            aggregate["weighted_rmse_relative_reduction"],
            1.0 - math.sqrt(4.0 / 7.0),
        )

    def test_rejects_inconsistent_duplicate_computations(self) -> None:
        original = self.row(
            0,
            values=100,
            baseline_mse=4.0,
            candidate_mse=1.0,
            multiplier=2.0,
        )
        conflicting = {**original, "global_divisor_multiplier": 1.0}
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.jsonl"
            report.write_text(
                "\n".join(json.dumps(row) for row in (original, conflicting))
                + "\n"
            )
            with self.assertRaisesRegex(AssertionError, "inconsistent"):
                verify_reports([report], {(3, 0, "down_proj")})


if __name__ == "__main__":
    unittest.main()
