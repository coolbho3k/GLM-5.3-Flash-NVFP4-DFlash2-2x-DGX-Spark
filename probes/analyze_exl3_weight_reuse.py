#!/usr/bin/env python3
"""CPU-only accounting of packed EXL3 weight reads, not DRAM profiling.

The source checks are drift guards, not a proof of arbitrary CUDA programs.
Address enumeration mirrors the guarded K4/M16/N256, K32 serving kernel.
Existing synthetic routing records must not be labelled live model routing.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit_source(root):
    paths = [root / 'quant' / name for name in (
        'glm53_exl3_moe_fast_kernel.cuh', 'exl3_gemm_inner.cuh')]
    moe, gemm = [p.read_text() for p in paths]
    required_moe = {
        'single_group_per_expert':
            'if (expert_idx_assign++ % concurrency != group_idx) continue;',
        'gate_up_m16_loop': 'in_addr += 16 * hidden_dim;',
        'down_m16_loop': 'in_addr += 16 * intermediate_dim;',
    }
    required_gemm = {
        'm16_only': 'static_assert(TILESIZE_M == 16, "Invalid kernel params");',
        'disjoint_slice_begin':
            'int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;',
        'disjoint_slice_end':
            'int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;',
        'one_copy_subgroup':
            'if (sub_k) { cp_async_fence(); return; }',
        'weight_copy_address':
            'if (pred_b_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i]);',
        'weight_stride':
            'load_b_gl[i] = k * (blocks_n * 256 / 16 * bits / 8) + n;',
    }
    for source, anchors in ((moe, required_moe), (gemm, required_gemm)):
        compact = ''.join(source.split())
        for name, anchor in anchors.items():
            if compact.count(''.join(anchor.split())) != 1:
                raise ValueError(f'Source drift: {name}')
    if moe.count('MIN(size_m, 16)') != 2:
        raise ValueError('Source drift: both GEMMs must iterate M16')
    return {'scope': 'structural guards, not formal CUDA verification',
            'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in paths},
            'guards': list(required_moe) + list(required_gemm)}


def enumerate_weight_copies(k, n, groups=8):
    """Count every int4 B address for ONE M16 expert/projection pass.

    No tensor or GPU allocations. The model includes the sub_k==0 guard;
    the other 256 threads do not issue a second copy of these addresses.
    """
    if k <= 0 or k % 32 or n <= 0 or n % 256 or groups <= 0:
        raise ValueError('Requires positive K32/N256-aligned dimensions')
    tiles_k, tiles_n, blocks_n = k // 32, n // 256, n // 16
    # 2 K16 blocks * 16 N16 blocks * 256 elements * 4 bits / 128 bits.
    chunks_per_tile = 256
    addresses = Counter()
    for block in range(groups):
        start = tiles_k * tiles_n * block // groups
        end = tiles_k * tiles_n * (block + 1) // groups
        for tile in range(start, end):
            kt, nt = tile % tiles_k, tile // tiles_k
            base = kt * (blocks_n * 2 * 64) // 8 + nt * (16 * 64) // 8
            for t in range(chunks_per_tile):
                local_n, local_k = t % 128, t // 128
                addresses[base + local_k * (blocks_n * 64 // 8) + local_n] += 1
    expected = k * n // 32  # 4 bits/element, 16 bytes/copy.
    if set(addresses) != set(range(expected)):
        raise ValueError('Address model has gaps or out-of-range copies')
    return {'k': k, 'n': n, 'blocks_per_expert': groups,
            'unique_bytes': len(addresses) * 16,
            'issued_copy_bytes': sum(addresses.values()) * 16,
            'max_copies_per_address': max(addresses.values()),
            'duplicate_bytes': (sum(addresses.values()) - len(addresses)) * 16}


def account(counts, rows, hidden=4096, intermediate=1024):
    if type(rows) is not int or rows < 0 or hidden <= 0 or intermediate <= 0:
        raise ValueError('Invalid dimensions')
    if any(type(c) is not int or not 0 <= c <= rows for c in counts):
        raise ValueError('Each expert must have at most one route per input row')
    if rows > 128:
        raise ValueError('This report models the thin path, not fat fallback')
    active = sum(c > 0 for c in counts)
    tiles = sum((c + 15) // 16 for c in counts)
    per_expert = 3 * hidden * intermediate // 2
    # M32/M64 are hypothetical shared-weight kernels, not existing fast paths.
    alternatives = {}
    for tile_m in (32, 64):
        candidate = sum((c + tile_m - 1) // tile_m for c in counts)
        alternatives[str(tile_m)] = {
            'weight_passes': candidate,
            'copy_byte_reduction_pct': 100 * (tiles - candidate) / tiles if tiles else 0,
        }
    return {'rows': rows, 'routes': sum(counts), 'active_experts': active,
            'max_rows_per_expert': max(counts, default=0),
            'm16_weight_passes': tiles, 'extra_weight_passes': tiles - active,
            'issued_packed_weight_bytes': tiles * per_expert,
            'unique_packed_weight_bytes': active * per_expert,
            'ideal_copy_byte_reduction_pct': 100 * (tiles - active) / tiles if tiles else 0,
            'hypothetical_tiles': alternatives}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--synthetic-samples', type=Path, nargs='*', default=[])
    args = p.parse_args()
    result = {'measurement_kind': 'CPU source/address and synthetic-route accounting',
              'caveats': ['Not measured DRAM bytes or serving speedup',
                          'No real-model routing trace was collected',
                          'Cross-step cache residency is not modelled'],
              'c1_c2_k7': 'At most 8/16 rows: one M16 pass per active expert; no extra passes',
              'address_coverage': [enumerate_weight_copies(k, n) for k, n in (
                  (4096, 1024), (1024, 4096))], 'synthetic_samples': []}
    if args.source:
        result['installed_source'] = audit_source(args.source)
    for path in args.synthetic_samples:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            counts = row['expert_counts']
            if len(counts) != 289 or counts[-1] != 0:
                raise ValueError('Expected 288 experts and an empty sentinel')
            report = account(counts[:-1], row['rows'])
            expected_routes = 0 if row['routing'] == 'empty' else row['rows'] * 8
            if report['routes'] != expected_routes:
                raise ValueError('Incomplete route counts')
            if report['m16_weight_passes'] != row['expert_m16_tiles']:
                raise ValueError('Existing tile count disagrees with accounting')
            result['synthetic_samples'].append({
                'source': str(path), 'routing': row['routing'], **report})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
