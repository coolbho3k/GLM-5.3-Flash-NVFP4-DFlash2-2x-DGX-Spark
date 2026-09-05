#!/usr/bin/env python3
"""Summarize isolated scheduling results and an explicitly simplified cost model."""
import argparse
import heapq
import json
from pathlib import Path


def schedule_model(counts, groups=6, cap=128):
    # Counts over cap are handled outside the fused kernel. M16 tiles are a
    # proxy for packed-weight GEMM work, not a measured per-expert duration.
    costs = [(n + 15)//16 for n in counts if 0 < n <= cap]
    static = [0] * groups
    queue = [(0, g) for g in range(groups)]
    heapq.heapify(queue)
    for i, cost in enumerate(costs):
        static[i % groups] += cost
        time, group = heapq.heappop(queue)
        heapq.heappush(queue, (time + cost, group))
    return dict(round_robin_group_tiles=static,
                round_robin_makespan=max(static),
                greedy_makespan=max(t for t, _ in queue),
                optimistic_lower_bound=max(max(costs, default=0), sum(costs)/groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.results:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if 'variants' not in row:
                continue
            control = row['variants']['g8_shared_k32_f1_s8']
            candidate = row['variants']['g8_shared_k32_f1_s8_tickets']
            print(json.dumps(dict(file=path.name, rows=row['rows'], routing=row['routing'],
                control_ms=control['ms'], ticket_ms=candidate['ms'],
                ticket_time_change_pct=100*(candidate['ms']/control['ms'] - 1),
                relative_output_rmse=candidate['relative_rmse'],
                tile_cost_model=schedule_model(row['expert_counts']))))


if __name__ == '__main__':
    main()
