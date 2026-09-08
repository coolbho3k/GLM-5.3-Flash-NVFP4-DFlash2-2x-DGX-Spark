#!/usr/bin/env python3
"""Matched sampled-decode screen; controls proposal mode at server launch.

Run against one server at a time. Warmups are excluded. Expected request counts
and preemptions are checked; differing output text is not a quality metric.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import statistics
import time

from bench_serving_quick import PROMPTS, metrics
from bench_prefill_mixed import health, stream_chat


def payload(kind, tokens, temperature, top_p, seed):
    if not 0 <= temperature <= 2 or not 0 < top_p <= 1 or tokens <= 0:
        raise ValueError('Invalid sampling parameters')
    return {'model': 'glm-5.3-flash', 'messages': [
        {'role': 'user', 'content': PROMPTS[kind]}],
        'temperature': temperature, 'top_p': top_p, 'top_k': -1,
        'presence_penalty': 0, 'frequency_penalty': 0, 'repetition_penalty': 1,
        'seed': seed, 'max_tokens': tokens, 'ignore_eos': True,
        'chat_template_kwargs': {'enable_thinking': False}}


def wave(url, requests):
    before = metrics(url)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as pool:
        streams = list(pool.map(lambda p: stream_chat(url, p, timeout=180), requests))
    wall = max(s['finished_monotonic'] for s in streams) - started
    after = metrics(url)
    delta = {key: after[key] - before[key] for key in before}
    if delta['request_success_total'] != len(requests) or delta['num_preemptions_total']:
        raise RuntimeError(f'Contaminated wave: {delta}')
    if any(s['completion_tokens'] != p['max_tokens'] for s, p in zip(streams, requests)):
        raise RuntimeError('Incomplete generation')
    drafts = delta['spec_decode_num_drafts_total']
    proposed = delta['spec_decode_num_draft_tokens_total']
    if drafts <= 0 or proposed <= 0:
        raise RuntimeError('No speculative verification observed')
    return {'wall_seconds': wall, 'streams': streams, 'metrics_delta': delta,
            'aggregate_tps': sum(s['completion_tokens'] for s in streams) / wall,
            'accepted_fraction': delta['spec_decode_num_accepted_tokens_total'] / proposed if proposed else None,
            'accepted_per_draft': delta['spec_decode_num_accepted_tokens_total'] / drafts if drafts else None,
            'c1_wall_ms_per_draft': wall * 1000 / drafts if len(requests) == 1 and drafts else None}


def compare(a, b):
    for key in ('levels', 'rounds', 'tokens', 'temperature', 'top_p', 'seed', 'warm_tokens'):
        if a['config'][key] != b['config'][key]:
            raise ValueError(f'Mismatched {key}')
    index = lambda r: {(c['round'], c['concurrency'], c['kind']): c for c in r['cases']}
    ai, bi = index(a), index(b)
    expected = len(a['config']['levels'].split(',')) * 2 * a['config']['rounds']
    if ai.keys() != bi.keys() or len(ai) != expected or len(a['cases']) != expected or len(b['cases']) != expected:
        raise ValueError('Incomplete/duplicate cases')
    grouped = {}
    for key in ai:
        old, new = ai[key], bi[key]
        if old['requests'] != new['requests']:
            raise ValueError('Mismatched requests')
        for c in (old, new):
            if c['metrics_delta']['request_success_total'] != key[1] or c['metrics_delta']['num_preemptions_total']:
                raise ValueError('Contaminated traffic')
        grouped.setdefault(key[1:], []).append((old, new))
    result = []
    for (concurrency, kind), pairs in sorted(grouped.items()):
        summaries = []
        for side in (0, 1):
            cases = [pair[side] for pair in pairs]
            summaries.append({key: statistics.mean(c[key] for c in cases)
                              for key in ('aggregate_tps', 'accepted_fraction', 'accepted_per_draft')})
            if concurrency == 1:
                summaries[-1]['wall_ms_per_draft'] = statistics.mean(c['c1_wall_ms_per_draft'] for c in cases)
        result.append({'concurrency': concurrency, 'kind': kind,
                       'baseline': summaries[0], 'candidate': summaries[1],
                       'gain_percent': 100 * (summaries[1]['aggregate_tps'] / summaries[0]['aggregate_tps'] - 1)})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--label', default='sampling-screen')
    p.add_argument('--output', type=Path)
    p.add_argument('--levels', default='1,3,6')
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--tokens', type=int, default=256)
    p.add_argument('--warm-tokens', type=int, default=64)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--seed', type=int, default=7301)
    p.add_argument('--compare', type=Path, nargs=2)
    args = p.parse_args()
    if args.compare:
        print(json.dumps(compare(*(json.loads(f.read_text()) for f in args.compare)), indent=2))
        return
    if args.output is None or args.rounds < 1:
        p.error('--output and positive rounds required')
    levels = [int(c) for c in args.levels.split(',')]
    if len(set(levels)) != len(levels) or any(c not in range(1, 7) for c in levels):
        p.error('Unique levels in 1..6 required')
    health(args.url)
    report = {'config': {**vars(args), 'output': str(args.output)}, 'cases': [], 'warmups': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for round_index in range(args.rounds):
        for c in levels:
            for kind in PROMPTS:
                requests = [payload(kind, args.tokens, args.temperature, args.top_p,
                                    args.seed + 100 * round_index + i) for i in range(c)]
                if round_index == 0 and args.warm_tokens:
                    warm = [{**r, 'max_tokens': args.warm_tokens} for r in requests]
                    report['warmups'].append({'concurrency': c, 'kind': kind, **wave(args.url, warm)})
                result = {'round': round_index, 'concurrency': c, 'kind': kind,
                          'requests': requests, **wave(args.url, requests)}
                report['cases'].append(result)
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps({k: v for k, v in result.items() if k not in ('requests', 'streams')}), flush=True)


if __name__ == '__main__':
    main()
