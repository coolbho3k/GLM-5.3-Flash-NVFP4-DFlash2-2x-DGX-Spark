#!/usr/bin/env python3
"""Check timing accounting independent of when the caller joins a wave."""
import unittest
from unittest.mock import patch

from bench_scheduler_pressure import finish_decode_wave
from summarize_torch_trace import union_duration


class AccountingTests(unittest.TestCase):
    def test_prefill_wait_does_not_extend_finished_decoder_wave(self):
        streams = [dict(finished_monotonic=end, decode_tokens_per_second=10,
                        max_stream_gap_seconds=0.2, p95_stream_gap_seconds=0.1,
                        completion_tokens=100, ttft_seconds=0.2)
                   for end in (19, 20)]
        with patch("time.perf_counter", return_value=1000):
            result = finish_decode_wave([], [{"result": s} for s in streams], 10)
        self.assertEqual(result["wall_seconds"], 10)
        self.assertEqual(result["aggregate_tokens_per_second"], 20)

    def test_overlapping_gpu_streams_are_not_double_counted(self):
        self.assertEqual(union_duration([(0, 3), (1, 2), (2, 5), (7, 8)]), 6)
        self.assertEqual(union_duration([]), 0)


if __name__ == "__main__":
    unittest.main()
