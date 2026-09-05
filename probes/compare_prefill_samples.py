#!/usr/bin/env python3
"""Compare single-pass fixed-text cold-prefill samples with traffic checks."""
import argparse
import json
from pathlib import Path
import statistics


def compare(baseline, candidate):
    groups = {'baseline': baseline, 'candidate': candidate}
    hashes, lengths, salts = set(), set(), set()
    result = {}
    count = 0
    for mode, files in groups.items():
        samples = []
        for path in files:
            report = json.loads(path.read_text())
            assert report['config']['mode'] == 'prefill' and report['config']['stable_prefill']
            assert not report['config'].get('profile')
            assert len(report['cases']) == 1
            case = report['cases'][0]
            assert case['global_prefix_hit_tokens_delta'] == 0
            assert case['content_preview'].strip() == 'OK'
            assert case['cache_salt']
            metrics = case['metrics_delta']
            assert metrics['request_success_total'] == 1, (path, 'unexpected completed requests')
            assert metrics['num_preemptions_total'] == 0
            hashes.add(case['prompt_sha256'])
            lengths.add(case['prompt_tokens'])
            salts.add(case['cache_salt'])
            count += 1
            samples.append({'file': str(path), 'input_tps': case['input_tokens_per_second']})
        assert samples
        rates = [s['input_tps'] for s in samples]
        result[mode] = {'samples': samples, 'median_input_tps': statistics.median(rates),
                        'mean_input_tps': statistics.mean(rates),
                        'min_input_tps': min(rates), 'max_input_tps': max(rates)}
    assert len(hashes) == len(lengths) == 1 and len(salts) == count
    result['prompt_tokens'] = lengths.pop()
    result['gain_percent'] = 100 * (result['candidate']['median_input_tps'] /
                                  result['baseline']['median_input_tps'] - 1)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', nargs='+', type=Path, required=True)
    parser.add_argument('--candidate', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.candidate), indent=2))
