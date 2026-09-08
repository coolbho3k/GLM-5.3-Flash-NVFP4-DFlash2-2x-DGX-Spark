#!/usr/bin/env python3
"""Paired-seed C1 confirmation; intervals condition on these prompts/boots.

Use a fixed, preselected seed set and sample count, not repeated optional
testing until an interval turns positive. Throughput is total output / time.
"""
import argparse
import json
from pathlib import Path
import random

from bench_draft_sampling import compare


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def summarize_pairs(pairs, resamples=10000):
    if len(pairs) < 4 or resamples < 100:
        raise ValueError('At least four paired seeds and 100 resamples required')
    durations = [(a['wall_seconds'], b['wall_seconds']) for a, b in pairs]
    outputs = [sum(s['completion_tokens'] for s in a['streams']) for a, _ in pairs]
    if len(set(outputs)) != 1 or any(min(a, b) <= 0 for a, b in durations):
        raise ValueError('Requires equal output counts and positive timings')
    def gain(selected):
        return 100 * (sum(durations[i][0] for i in selected) /
                      sum(durations[i][1] for i in selected) - 1)
    rng = random.Random(5307)
    n = len(pairs)
    gains = [gain([rng.randrange(n) for _ in range(n)]) for _ in range(resamples)]
    modes = []
    for side in (0, 1):
        cases = [p[side] for p in pairs]
        drafts = sum(c['metrics_delta']['spec_decode_num_drafts_total'] for c in cases)
        proposed = sum(c['metrics_delta']['spec_decode_num_draft_tokens_total'] for c in cases)
        accepted = sum(c['metrics_delta']['spec_decode_num_accepted_tokens_total'] for c in cases)
        modes.append({'total_output_tokens': sum(outputs),
                      'total_wall_seconds': sum(c['wall_seconds'] for c in cases),
                      'throughput_tps': sum(outputs) / sum(c['wall_seconds'] for c in cases),
                      'case_tps_samples': [c['aggregate_tps'] for c in cases],
                      'accepted_fraction': accepted / proposed,
                      'accepted_per_draft': accepted / drafts,
                      'wall_ms_per_draft': 1000 * sum(c['wall_seconds'] for c in cases) / drafts})
    lower, upper = percentile(gains, 0.025), percentile(gains, 0.975)
    return {'paired_seeds': n, 'baseline': modes[0], 'candidate': modes[1],
            'throughput_gain_percent': gain(range(n)),
            'paired_seed_bootstrap_95_percent': [lower, upper],
            'candidate_faster_pairs': sum(b < a for a, b in durations),
            'confirmed_for_this_prompt': lower > 0}


def confirmation(a, b):
    compare(a, b)  # Existing request/config/completeness/traffic guards.
    if a['config']['levels'] != '1' or a['config']['rounds'] < 4:
        raise ValueError('Confirmation requires C1 and >=4 paired seeds')
    index = lambda r: {(c['round'], c['kind']): c for c in r['cases']}
    ai, bi = index(a), index(b)
    kinds = sorted({key[1] for key in ai})
    by_kind = {}
    for kind in kinds:
        pairs = [(ai[(i, kind)], bi[(i, kind)]) for i in range(a['config']['rounds'])]
        for pair in pairs:
            for case in pair:
                if len(case['streams']) != 1 or case['streams'][0]['completion_tokens'] != a['config']['tokens']:
                    raise ValueError('Incomplete or non-C1 generation')
        by_kind[kind] = summarize_pairs(pairs)
    return {'schema': 1, 'temperature': a['config']['temperature'],
            'top_p': a['config']['top_p'], 'tokens_per_request': a['config']['tokens'],
            'seed_base': a['config']['seed'], 'by_prompt': by_kind,
            'confirmed_both_prompts': all(r['confirmed_for_this_prompt'] for r in by_kind.values()),
            'caveats': ['Intervals quantify seed variation on these two fixed prompts only.',
                        'One boot per mode: restart/thermal variability is not estimated.',
                        'Different continuations are expected; this is not a quality evaluation.',
                        'Temperature 1.0, thinking disabled; no claim for other traffic.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    args = p.parse_args()
    print(json.dumps(confirmation(*(json.loads(f.read_text()) for f in
                                   (args.baseline, args.candidate))), indent=2))


if __name__ == '__main__':
    main()
