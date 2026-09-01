#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = load("capture_accuracy_signature")
compare = load("compare_accuracy_signatures")


class AccuracySignatureTests(unittest.TestCase):
    def test_metric_value_sums_labeled_engines(self) -> None:
        text = "metric{engine=\"0\"} 3\nmetric{engine=\"1\"} 4\n"
        self.assertEqual(capture.metric_value(text, "metric"), 7)

    def test_subtract_acceptance(self) -> None:
        before = {
            "drafts": 10,
            "draft_tokens": 70,
            "accepted_tokens": 21,
            "accepted_per_position": {"0": 8},
        }
        after = {
            "drafts": 12,
            "draft_tokens": 84,
            "accepted_tokens": 28,
            "accepted_per_position": {"0": 10, "1": 3},
        }
        result = capture.subtract(after, before)
        self.assertEqual(result["acceptance_ratio"], 0.5)
        self.assertEqual(result["mean_acceptance_length"], 4.5)
        self.assertEqual(result["accepted_per_position"], {"0": 2, "1": 3})

    def test_js_identity_and_shift(self) -> None:
        same = {"a": 0.75, "b": 0.25}
        self.assertAlmostEqual(compare.js_divergence(same, same), 0.0)
        shifted = compare.js_divergence(same, {"a": 0.25, "b": 0.75})
        self.assertGreater(shifted, 0.1)

    def test_paired_positions_stops_at_first_divergence(self) -> None:
        def row(token: str):
            return {
                "token_key": token,
                "top_logprobs": {token: -0.1, "other": -2.4},
            }

        positions = compare.paired_positions(
            [row("a"), row("b"), row("c")],
            [row("a"), row("x"), row("c")],
        )
        self.assertEqual(len(positions), 2)
        self.assertTrue(positions[0]["same_sampled_token"])
        self.assertFalse(positions[1]["same_sampled_token"])

    def test_top1_ignores_aggregated_topk_tail(self) -> None:
        student = {
            "token_key": "a",
            "top_logprobs": {"a": -2.0, "b": -2.1},
        }
        teacher = {
            "token_key": "b",
            "top_logprobs": {"a": -2.1, "b": -2.0},
        }
        self.assertGreater(
            compare.distribution(student)["__TOPK_TAIL__"],
            max(compare.distribution(student)[key] for key in ("a", "b")),
        )
        positions = compare.paired_positions([student], [teacher])
        self.assertEqual(len(positions), 1)
        self.assertFalse(positions[0]["same_top1"])


if __name__ == "__main__":
    unittest.main()
