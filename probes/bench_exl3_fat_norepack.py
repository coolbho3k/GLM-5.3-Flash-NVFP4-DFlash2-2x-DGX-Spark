#!/usr/bin/env python3
"""Isolated no-repack parity/timing. Serving must be stopped before running.

Compare full gate/up staging + M64 GEMM against separate-buffer M64, and
report kernel-only controls separately. Inputs and math are unchanged.
"""
import argparse
import ctypes as C
import json
from pathlib import Path

import torch
import exllamav3_ext as ext
from bench_exl3_fat_pipeline import timed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libraries', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--rows', default='129,145,256,512,1024,2048,8192')
    parser.add_argument('--small-only', action='store_true')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--native-norepack', action='store_true')
    args = parser.parse_args()
    assert args.repeats > 0 and ext.glm53_fat_pipeline_version() == 2
    if args.native_norepack:
        assert ext.glm53_fat_norepack_version() == 1
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.04)
    torch.manual_seed(73)
    libraries = {}
    for name in ('m64_control', 'm64_norepack'):
        lib = C.CDLL(str((args.libraries / (name + '.so')).resolve()))
        lib.fat_probe_launch.restype = C.c_int
        lib.fat_probe_launch.argtypes = [C.POINTER(C.c_void_p), *([C.c_int]*4), C.c_void_p]
        lib.fat_probe_occupancy.argtypes = [C.c_int]
        libraries[name] = lib
    shapes = [(1, 16, 256), (63, 32, 256), (64, 48, 512),
              (65, 48, 512), (127, 16, 256), (129, 48, 512)]
    if not args.small_only:
        shapes += [(int(m), 4096, 2048) for m in args.rows.split(',')]
    with torch.inference_mode():
        for m, k, n in shapes:
            assert 0 < m <= 8192 and n % 256 == 0
            a = torch.randn(m, k, dtype=torch.float16, device='cuda') * 0.2
            gate, up = [torch.randint(-32768, 32767, (k//16, n//32, 64),
                        dtype=torch.int16, device='cuda') for _ in range(2)]
            sg, su = [torch.randn(n//2, dtype=torch.float16, device='cuda') * 0.03 for _ in range(2)]
            packed = torch.empty(k//16, n//16, 64, dtype=torch.int16, device='cuda')
            svh = torch.empty(n, dtype=torch.float16, device='cuda')
            out = torch.empty(m, n, dtype=torch.float32, device='cuda')
            rejected_inputs = 0
            if args.native_norepack and (m, k, n) == shapes[0]:
                invalid_calls = [
                    lambda: ext.exl3_fat_gate_up_norepack(a, gate, up, out, sg[:-1], su, 4, True, False),
                    lambda: ext.exl3_fat_gate_up_norepack(a, gate, up[:, :-1], out, sg, su, 4, True, False),
                    lambda: ext.exl3_fat_gate_up_norepack(a, gate, up, out.half(), sg, su, 4, True, False),
                    lambda: ext.exl3_fat_gate_up_norepack(a, gate, up, out, sg, su, 8, True, False),
                    lambda: ext.exl3_fat_gate_up_norepack(a[:0], gate, up, out[:0], sg, su, 4, True, False),
                ]
                for invalid in invalid_calls:
                    try:
                        invalid()
                    except RuntimeError:
                        rejected_inputs += 1
                    else:
                        raise AssertionError('Native wrapper accepted invalid input')

            def repack():
                packed[:, :n//32].copy_(gate)
                packed[:, n//32:].copy_(up)
                svh[:n//2].copy_(sg)
                svh[n//2:].copy_(su)

            def native():
                ext.exl3_fat_gemm_async2_m64(a, packed, out, svh, 4, True, False)

            def native_repack():
                repack()
                native()

            repack()
            native()
            reference = out.clone()
            holders, argv = {}, {}
            for name in libraries:
                tensors = [a, gate, up, out, sg, su] if name == 'm64_norepack' else [a, packed, out, svh]
                values = [C.c_void_p(t.data_ptr()) for t in tensors]
                values += [C.c_void_p(0), C.c_void_p(0)]
                values += [C.c_int(v) for v in (m, k, n)]
                holders[name] = values
                argv[name] = (C.c_void_p * len(values))(*[C.addressof(v) for v in values])

            def launch(name):
                rc = libraries[name].fat_probe_launch(argv[name], m, k, n, 0,
                                                     torch.cuda.current_stream().cuda_stream)
                assert rc == 0, (name, rc)

            def control_repack():
                repack()
                launch('m64_control')

            variants = {'native_kernel': native, 'native_with_repack': native_repack,
                        'control_kernel': lambda: launch('m64_control'),
                        'control_with_repack': control_repack,
                        'norepack': lambda: launch('m64_norepack')}
            if args.native_norepack:
                variants['installed_norepack'] = lambda: ext.exl3_fat_gate_up_norepack(
                    a, gate, up, out, sg, su, 4, True, False)
            row = {'m': m, 'k': k, 'n': n, 'repeats': args.repeats,
                   'invalid_native_inputs_rejected': rejected_inputs,
                   'order': 'reverse' if args.reverse else 'forward',
                   'copy_read_write_bytes_removed': 2 * (packed.numel() * 2 + svh.numel() * 2),
                   'variants': {}}
            names = list(variants)
            if args.reverse:
                names.reverse()
            for name in names:
                out.fill_(float('nan'))
                variants[name]()
                torch.cuda.synchronize()
                assert torch.isfinite(out).all() and torch.equal(out, reference), (name, m, k, n)
                ms = timed(variants[name], args.repeats)
                assert torch.equal(out, reference), ('graph replay', name, m, k, n)
                row['variants'][name] = {'ms': ms, 'bitwise_equal': True,
                                         'replay_bitwise_equal': True}
            row['resources'] = {name: {'shared_bytes': lib.fat_probe_shared_bytes(),
                               'resident_blocks_per_sm': lib.fat_probe_occupancy(0)}
                                for name, lib in libraries.items()}
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
