#!/usr/bin/env python3
import unittest
from compare_serving_quick import summarize


class SummaryTests(unittest.TestCase):
    def test_acceptance_and_iteration_accounting(self):
        cases = []
        for tps, seconds, accepted, successes in [(20, 10, 100, 1), (25, 8, 120, 2)]:
            cases.append({'concurrency': 1, 'kind': 'code',
                'aggregate_tokens_per_second': tps, 'wall_seconds': seconds,
                'streams': [{'decode_tokens_per_second': tps + 1, 'content_sha256': 'abc'}],
                'metrics_delta': {'spec_decode_num_drafts_total': 50,
                    'spec_decode_num_draft_tokens_total': 350,
                    'spec_decode_num_accepted_tokens_total': accepted,
                    'request_success_total': successes}})
        out = summarize({'cases': cases})[(1, 'code')]
        self.assertEqual(out['aggregate_tps_median'], 22.5)
        self.assertAlmostEqual(out['accepted_fraction'], 220 / 700)
        self.assertEqual(out['accepted_per_draft'], 2.2)
        self.assertEqual(out['wall_ms_per_draft'], 180)
        self.assertTrue(out['unexpected_completed_requests'])


if __name__ == '__main__':
    unittest.main()
