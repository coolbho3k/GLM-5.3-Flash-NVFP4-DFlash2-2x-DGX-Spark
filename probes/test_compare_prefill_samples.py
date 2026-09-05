"""Exercise traffic rejection against the committed confirmation samples."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from compare_prefill_samples import compare

RESULTS_ROOT = Path(__file__).resolve().parents[1] / 'results/serving-20260904'

class PrefillComparisonTests(unittest.TestCase):
    baseline = [RESULTS_ROOT / f'fat-norepack-A{i}-prefill.json' for i in (3, 4)]
    candidate = [RESULTS_ROOT / f'fat-norepack-B{i}-prefill.json' for i in (3, 4)]

    def test_saved_confirmation(self):
        report = compare(self.baseline, self.candidate)
        self.assertEqual(report['prompt_tokens'], 21850)
        self.assertAlmostEqual(report['gain_percent'], 2.0426458651662927)

    def reject_mutation(self, mutate):
        original = Path.read_text
        target = self.candidate[-1]
        def changed(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path == target:
                report = json.loads(text)
                mutate(report['cases'][0])
                return json.dumps(report)
            return text
        with patch.object(Path, 'read_text', changed), self.assertRaises(AssertionError):
            compare(self.baseline, self.candidate)

    def test_extra_request_is_rejected(self):
        self.reject_mutation(lambda c: c['metrics_delta'].update(request_success_total=2))

    def test_cache_hits_are_rejected(self):
        self.reject_mutation(lambda c: c.update(global_prefix_hit_tokens_delta=128))

    def test_reused_salt_is_rejected(self):
        salt = json.loads(self.baseline[0].read_text())['cases'][0]['cache_salt']
        self.reject_mutation(lambda c: c.update(cache_salt=salt))


if __name__ == '__main__':
    unittest.main()
