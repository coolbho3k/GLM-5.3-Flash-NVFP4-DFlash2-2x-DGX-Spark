#!/usr/bin/env python3
"""Read-only, stdlib-only sampled comparison with repaired source-FP8 tensors.

Does not load Torch, touch CUDA, or materialize whole tensors. Differences
disprove exact FP8 passthrough for the sample; equality cannot prove it for
the full checkpoint. This is a provenance check, not a quality benchmark.
"""
import argparse
import json
import math
import struct
from pathlib import Path

from repair_redhat_fp8_passthrough import read_safetensors_header


def fp8(byte):
    sign = -1 if byte & 128 else 1
    exponent, mantissa = (byte & 127) >> 3, byte & 7
    if exponent == 15 and mantissa == 7:
        return float('nan')
    return sign * (mantissa * 2**-9 if exponent == 0 else
                   (1 + mantissa / 8) * 2**(exponent - 7))


def bf16_bits(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / 'model.safetensors.index.json').read_text())['weight_map']
        self.headers = {}

    def entry(self, name):
        shard = self.index[name]
        if shard not in self.headers:
            self.headers[shard] = read_safetensors_header(self.root / shard)
        header, base = self.headers[shard]
        return header[name], self.root / shard, base

    def read(self, name, offset, count):
        entry, path, base = self.entry(name)
        begin, end = entry['data_offsets']
        assert 0 <= offset <= offset + count <= end - begin
        with path.open('rb') as handle:
            handle.seek(base + begin + offset)
            result = handle.read(count)
        assert len(result) == count
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exl3', required=True)
    parser.add_argument('--fp8-repaired', required=True)
    args = parser.parse_args()
    assert fp8(0x38) == 1 and fp8(0x7e) == 448 and fp8(1) == 2**-9
    assert bf16_bits(1.0) == 0x3f80
    exl, source = Checkpoint(args.exl3), Checkpoint(args.fp8_repaired)
    names = ['0.mlp.down_proj', '3.mlp.shared_experts.down_proj',
             '3.self_attn.q_b_proj', '0.self_attn.q_proj', '0.self_attn.o_proj']
    for suffix in names:
        name = 'model.language_model.layers.' + suffix + '.weight'
        target_entry = exl.entry(name)[0]
        source_entry = source.entry(name)[0]
        assert target_entry['dtype'] == 'BF16'
        assert target_entry['shape'] == source_entry['shape']
        n, k = target_entry['shape']
        stats = dict(name=name, shape=[n, k], exl3_dtype='BF16',
                     reference_dtype=source_entry['dtype'], samples=0, differing=0)
        error2 = value2 = 0.0
        for row in [0, n // 2, n - 1]:
            for column in [0, (k // 256) * 128, k - 128]:
                raw = exl.read(name, 2 * (row * k + column), 256)
                target_bits = struct.unpack('<128H', raw)
                if source_entry['dtype'] == 'BF16':
                    reference_bits = struct.unpack('<128H', source.read(name, 2 * (row * k + column), 256))
                else:
                    assert source_entry['dtype'] == 'F8_E4M3'
                    scale_name = name + '_scale'
                    scales = source.entry(scale_name)[0]
                    assert scales['dtype'] == 'F32' and scales['shape'] == [(n + 127)//128, (k + 127)//128]
                    scale_offset = 4 * ((row // 128) * scales['shape'][1] + column // 128)
                    scale = struct.unpack('<f', source.read(scale_name, scale_offset, 4))[0]
                    reference_bits = [bf16_bits(fp8(b) * scale) for b in source.read(name, row * k + column, 128)]
                for actual, reference in zip(target_bits, reference_bits):
                    a = struct.unpack('<f', struct.pack('<I', actual << 16))[0]
                    b = struct.unpack('<f', struct.pack('<I', reference << 16))[0]
                    assert math.isfinite(a) and math.isfinite(b)
                    stats['samples'] += 1
                    stats['differing'] += actual != reference
                    error2 += (a - b)**2
                    value2 += a*a
        stats['relative_rmse_to_exl3_sample'] = math.sqrt(error2 / value2) if value2 else None
        print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
