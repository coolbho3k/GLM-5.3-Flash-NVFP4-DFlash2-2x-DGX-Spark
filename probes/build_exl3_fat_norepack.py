#!/usr/bin/env python3
"""CPU-only isolated build: M64 gate/up reads separate immutable weight buffers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from build_exl3_fat_pipeline import pipelined_source, small_tile_source, standalone


def separate_gate_up_source(source):
    """Change input addressing only; each N tile belongs to exactly one half."""
    def once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"Expected one no-repack anchor: {old!r}")
        source = source.replace(old, new, 1)

    if 'FAT_TILE_M = 64;' not in source or 'auto prefetch = ' not in source:
        raise ValueError('No-repack screen requires the validated M64 pipeline')
    once('    const uint16_t* __restrict__ packed,',
         '    const uint16_t* __restrict__ packed_gate,\n'
         '    const uint16_t* __restrict__ packed_up,')
    once('    const half* __restrict__ svh,',
         '    const half* __restrict__ svh_gate,\n'
         '    const half* __restrict__ svh_up,')
    once('    int tiles_n = size_n / 16;', '''    // N/2 is a multiple of 128: no tile crosses the gate/up boundary.
    const int half_n = size_n / 2;
    const bool is_up = n_base >= half_n;
    const int packed_n_base = is_up ? n_base - half_n : n_base;
    const uint16_t* packed = is_up ? packed_up : packed_gate;
    const half* svh = is_up ? svh_up : svh_gate;
    int tiles_n = half_n / 16;''')
    once('packed + (k_block * tiles_n + n_base / 16) * FAT_PACKED_WORDS',
         'packed + (k_block * tiles_n + packed_n_base / 16) * FAT_PACKED_WORDS')
    once('                svh + n_base);', '                svh + packed_n_base);')
    return source


def generated_source(original, separate):
    source = small_tile_source(pipelined_source(original))
    if separate:
        source = separate_gate_up_source(source)
    result = standalone(source, 2)
    if separate:
        anchor = 'n <= 0 || n > 4096 || n % 128 || (scatter != 0 && scatter != 1)'
        if result.count(anchor) != 1:
            raise ValueError('Unexpected C ABI launch guard')
        result = result.replace(anchor, 'n <= 0 || n > 4096 || n % 256 || scatter != 0')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext')
    output = args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError('Never build into the installed extension')
    output.mkdir(parents=True, exist_ok=True)
    original = args.source.read_text()
    for name, separate in [('m64_control', False), ('m64_norepack', True)]:
        generated = generated_source(original, separate)
        path = output / (name + '.cu')
        path.write_text(generated)
        command = ['/usr/local/cuda/bin/nvcc', '-O3', '--use_fast_math', '-std=c++17',
                   '-arch=sm_121', '--shared', '-Xcompiler=-fPIC', '-Xlinker=-Bsymbolic',
                   '-Xptxas=-v', '-lineinfo', '-I', str(root), str(path),
                   '-o', str(output / (name + '.so'))]
        subprocess.run(command, check=True)
        print(json.dumps({'variant': name, 'command': command,
                         'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
                         'generated_sha256': hashlib.sha256(generated.encode()).hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
