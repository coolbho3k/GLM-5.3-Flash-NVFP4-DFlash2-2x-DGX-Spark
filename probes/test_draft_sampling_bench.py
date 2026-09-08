import unittest
from copy import deepcopy
from bench_draft_sampling import payload, compare


class SamplingBenchTests(unittest.TestCase):
    def test_explicit_matched_sampling(self):
        p = payload('code', 256, 0.7, 0.95, 42)
        self.assertEqual(p, payload('code', 256, 0.7, 0.95, 42))
        self.assertEqual(p['temperature'], 0.7)
        self.assertEqual(p['top_k'], -1)
        self.assertEqual(p['repetition_penalty'], 1)
        self.assertNotEqual(p, payload('code', 256, 0.7, 0.95, 43))

    def test_invalid_sampling(self):
        for t, top_p in ((-1, 1), (0.7, 0), (3, 1)):
            with self.assertRaises(ValueError):
                payload('prose', 256, t, top_p, 42)

    def test_partial_run_rejected(self):
        cfg = dict(levels='1', rounds=1, tokens=256, temperature=0.7,
                   top_p=0.95, seed=42, warm_tokens=64)
        with self.assertRaises(ValueError):
            compare({'config': cfg, 'cases': []}, {'config': cfg, 'cases': []})

    def fixture(self):
        return {'config': dict(levels='1', rounds=1, tokens=256, temperature=0.7,
                               top_p=0.95, seed=42, warm_tokens=64),
                'cases': [dict(round=0, concurrency=1, kind=kind,
                               requests=[payload(kind, 256, 0.7, 0.95, 42)],
                               metrics_delta={'request_success_total': 1, 'num_preemptions_total': 0},
                               aggregate_tps=30, accepted_fraction=0.4,
                               accepted_per_draft=2.8, c1_wall_ms_per_draft=120)
                          for kind in ('prose', 'code')]}

    def test_matched_comparison(self):
        a = self.fixture()
        b = deepcopy(a)
        for c in b['cases']:
            c['aggregate_tps'] = 36
        self.assertTrue(all(abs(r['gain_percent'] - 20) < 1e-9 for r in compare(a, b)))

    def test_contamination_and_sampling_drift_rejected(self):
        a = self.fixture()
        for mutate in (
            lambda b: b['cases'][0]['requests'][0].update(seed=43),
            lambda b: b['cases'][0]['metrics_delta'].update(request_success_total=2),
            lambda b: b['cases'][0]['metrics_delta'].update(num_preemptions_total=1),
            lambda b: b['cases'].append(deepcopy(b['cases'][0])),
        ):
            b = deepcopy(a)
            mutate(b)
            with self.assertRaises(ValueError):
                compare(a, b)


if __name__ == '__main__':
    unittest.main()
