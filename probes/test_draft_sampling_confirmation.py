import unittest
from confirm_draft_sampling import percentile, summarize_pairs


def case(seconds):
    return {'wall_seconds': seconds, 'aggregate_tps': 512 / seconds,
            'streams': [{'completion_tokens': 512}],
            'metrics_delta': {'spec_decode_num_drafts_total': 100,
                              'spec_decode_num_draft_tokens_total': 700,
                              'spec_decode_num_accepted_tokens_total': 412}}


class ConfirmationTests(unittest.TestCase):
    def test_known_gain(self):
        pairs = [(case(t), case(t / 1.1)) for t in (10, 11, 12, 13, 14, 15)]
        r = summarize_pairs(pairs, 1000)
        self.assertAlmostEqual(r['throughput_gain_percent'], 10)
        self.assertTrue(r['confirmed_for_this_prompt'])

    def test_no_gain_and_noisy_pairs_not_confirmed(self):
        for timings in ((10, 10, 10, 10), (8, 12, 8, 12)):
            r = summarize_pairs([(case(10), case(t)) for t in timings], 1000)
            self.assertFalse(r['confirmed_for_this_prompt'])

    def test_total_time_not_arithmetic_mean_tps(self):
        r = summarize_pairs([(case(10), case(5)), (case(10), case(15))] * 2, 1000)
        self.assertAlmostEqual(r['throughput_gain_percent'], 0)

    def test_reject_small_sample(self):
        with self.assertRaises(ValueError):
            summarize_pairs([(case(10), case(9))] * 2)

    def test_percentile_interpolation(self):
        self.assertEqual(percentile([0, 10], 0.25), 2.5)


if __name__ == '__main__':
    unittest.main()
