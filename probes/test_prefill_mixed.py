#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_prefill_mixed as bench
import compare_prefill_mixed as compare


class PrefillMixedToolingTests(unittest.TestCase):
    def test_prefill_corpora_are_deterministic_and_case_distinct(self) -> None:
        kwargs = {
            "target_tokens": 8192,
            "corpus_seed": "test-seed",
            "phrase_index": 0,
        }
        first = bench.prefill_messages(case_id="first", **kwargs)
        repeated = bench.prefill_messages(case_id="first", **kwargs)
        second = bench.prefill_messages(case_id="second", **kwargs)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first[0]["content"], second[0]["content"])
        self.assertTrue(first[0]["content"].startswith("Read this synthetic"))

    def test_larger_prefill_target_builds_a_larger_prompt(self) -> None:
        small = bench.prefill_messages(
            8192, case_id="small", corpus_seed="test", phrase_index=0
        )
        large = bench.prefill_messages(
            32768, case_id="large", corpus_seed="test", phrase_index=1
        )
        self.assertGreater(len(large[0]["content"]), len(small[0]["content"]))

    def test_decode_payload_forces_an_exact_generation_budget(self) -> None:
        payload = bench.decode_payload(
            384, case_id="decode", corpus_seed="test-seed"
        )
        self.assertEqual(payload["max_tokens"], 384)
        self.assertTrue(payload["ignore_eos"])
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])

    def test_geometric_mean(self) -> None:
        self.assertAlmostEqual(compare.geometric_mean([1.0, 4.0]), 2.0)
        with self.assertRaises(ValueError):
            compare.geometric_mean([])


if __name__ == "__main__":
    unittest.main()
