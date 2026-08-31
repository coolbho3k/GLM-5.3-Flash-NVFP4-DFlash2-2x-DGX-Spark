#!/usr/bin/env python3
"""Measure representative two-rank DCP collectives over the production fabric."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarize(name: str, operation: str, payload_bytes: int,
              times_s: list[float], volume_factor: float) -> dict[str, object]:
    local = torch.tensor(
        [
            statistics.mean(times_s),
            percentile(times_s, 0.50),
            percentile(times_s, 0.95),
            min(times_s),
            max(times_s),
        ],
        dtype=torch.float64,
        device="cuda",
    )
    # The slowest rank determines the wall time seen by the model.
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    mean_s, p50_s, p95_s, min_s, max_s = local.cpu().tolist()
    bus_bytes = payload_bytes * volume_factor
    return {
        "name": name,
        "operation": operation,
        "payload_bytes_per_rank": payload_bytes,
        "mean_us": round(mean_s * 1e6, 3),
        "p50_us": round(p50_s * 1e6, 3),
        "p95_us": round(p95_s * 1e6, 3),
        "min_us": round(min_s * 1e6, 3),
        "max_us": round(max_s * 1e6, 3),
        "effective_bus_gib_s_at_mean": round(bus_bytes / mean_s / (1024**3), 3),
    }


def measure(
    *, name: str, operation: str, numel: int, dtype: torch.dtype,
    warmup: int, iterations: int, world_size: int,
) -> dict[str, object]:
    source = torch.arange(numel, dtype=dtype, device="cuda")
    if operation == "all_gather":
        destination = torch.empty(numel * world_size, dtype=dtype, device="cuda")

        def collective() -> None:
            dist.all_gather_into_tensor(destination, source)

        volume_factor = world_size - 1
    elif operation == "all_to_all":
        destination = torch.empty_like(source)

        def collective() -> None:
            dist.all_to_all_single(destination, source)

        volume_factor = (world_size - 1) / world_size
    elif operation == "all_reduce":
        destination = source.clone()

        def collective() -> None:
            dist.all_reduce(destination)

        volume_factor = 2 * (world_size - 1) / world_size
    else:
        raise ValueError(operation)

    dist.barrier()
    for _ in range(warmup):
        collective()
    torch.cuda.synchronize()
    dist.barrier()

    times_s: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        collective()
        torch.cuda.synchronize()
        times_s.append(time.perf_counter() - start)
    dist.barrier()

    return summarize(
        name,
        operation,
        source.numel() * source.element_size(),
        times_s,
        volume_factor,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-addr", default="10.100.32.1")
    parser.add_argument("--master-port", type=int, default=29631)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        rank=args.rank,
        world_size=args.world_size,
        timeout=timedelta(minutes=5),
    )
    try:
        cases = [
            # Sparse-indexer candidate scores/IDs and DCP query/output-sized
            # traffic. These are communication microbenchmarks, not claims that
            # every decode step issues each exact PyTorch collective.
            ("indexer_candidates_96KiB", "all_gather", 24_576, torch.float32),
            ("dcp_query_1.5MiB", "all_to_all", 786_432, torch.bfloat16),
            ("dcp_output_1.5MiB", "all_reduce", 786_432, torch.bfloat16),
        ]
        results = [
            measure(
                name=name,
                operation=operation,
                numel=numel,
                dtype=dtype,
                warmup=args.warmup,
                iterations=args.iterations,
                world_size=args.world_size,
            )
            for name, operation, numel, dtype in cases
        ]
        if args.rank == 0:
            print(
                json.dumps(
                    {
                        "schema": 1,
                        "backend": dist.get_backend(),
                        "world_size": args.world_size,
                        "gpu": torch.cuda.get_device_name(0),
                        "nccl_ib_hca": os.environ.get("NCCL_IB_HCA"),
                        "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
                        "warmup": args.warmup,
                        "iterations": args.iterations,
                        "results": results,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
