#!/usr/bin/env python3
"""Summarize paired quick decode reports, retaining acceptance/traffic caveats."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(report):
    grouped = defaultdict(list)
    for case in report['cases']:
        grouped[(case['concurrency'], case['kind'])].append(case)
    result = {}
    for key, cases in grouped.items():
        drafts = sum(c['metrics_delta']['spec_decode_num_drafts_total'] for c in cases)
        proposed = sum(c['metrics_delta']['spec_decode_num_draft_tokens_total'] for c in cases)
        accepted = sum(c['metrics_delta']['spec_decode_num_accepted_tokens_total'] for c in cases)
        result[key] = {
            'rounds': len(cases),
            'aggregate_tps_median': statistics.median(c['aggregate_tokens_per_second'] for c in cases),
            'aggregate_tps_samples': [c['aggregate_tokens_per_second'] for c in cases],
            'per_stream_decode_tps_median': statistics.median(
                s['decode_tokens_per_second'] for c in cases for s in c['streams']),
            'accepted_fraction': accepted / proposed if proposed else None,
            'accepted_per_draft': accepted / drafts if drafts else None,
            'completion_hashes': sorted({s['content_sha256'] for c in cases for s in c['streams']}),
            'unexpected_completed_requests': any(
                c['metrics_delta'].get('request_success_total', key[0]) != key[0] for c in cases),
        }
        if key[0] == 1 and drafts:
            # Diagnostic only: includes first-token work and is not a GPU
            # kernel timer. Helps identify an acceptance-driven speed change.
            result[key]['wall_ms_per_draft'] = 1000 * sum(c['wall_seconds'] for c in cases) / drafts
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    a, b = [json.loads(p.read_text()) for p in (args.baseline, args.candidate)]
    for field in ('mode', 'levels', 'prompts', 'max_tokens', 'rounds', 'profile'):
        assert a['config'].get(field) == b['config'].get(field), f'Mismatched {field}'
    assert a['config']['mode'] == 'decode'
    sa, sb = summarize(a), summarize(b)
    assert sa.keys() == sb.keys(), 'Incomplete or mismatched case sets'
    for key in sorted(sa):
        old, new = sa[key], sb[key]
        assert old['rounds'] == new['rounds'] == a['config']['rounds'], 'Incomplete rounds'
        print(json.dumps({'concurrency': key[0], 'kind': key[1],
            'aggregate_gain_percent': 100 * (new['aggregate_tps_median'] / old['aggregate_tps_median'] - 1),
            'baseline': old, 'candidate': new}), flush=True)


if __name__ == '__main__':
    main()
