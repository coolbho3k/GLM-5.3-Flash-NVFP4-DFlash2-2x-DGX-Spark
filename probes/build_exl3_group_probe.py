#!/usr/bin/env python3
"""Build isolated variants from installed EXL3 sources; no serving mutations.

Output must be an experiment directory, never the installed extension tree.
Run inside the serving image (nvcc required), without allocating a GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def private_output_source(kernel, hadamard):
    """Derive a scratch-private writeback; input must already reuse gate SUH.

    Each group processes experts sequentially, and top-k contains no duplicate
    expert per token. Its final group barrier orders successive += writes.
    The otherwise unused FP16 up-input buffer holds half as many FP32 rows.
    This is probe-only, not a patch to the installed serving extension.
    """
    def once(source, old, new):
        if source.count(old) != 1:
            raise ValueError(f'Expected one private-output anchor: {old!r}')
        return source.replace(old, new, 1)

    if 'gemm_up(temp_state_g, temp_intermediate_u' not in kernel:
        raise ValueError('Private output requires unconditional shared gate/up input')
    start = hadamard.index('inline __device__\nvoid had_hf_r_128_d_inner\n')
    end = hadamard.index('\n}', start) + 2
    helper = hadamard[start:end]
    helper = once(helper, 'void had_hf_r_128_d_inner', 'void glm53_private_had_d')
    for offset in (0, 32, 64, 96):
        helper = once(helper,
            f'atomicAdd(output_ptr + {offset:2d} + t, sh[{offset:2d} + t]);',
            f'output_ptr[{offset} + t] += sh[{offset} + t];')
    kernel = once(kernel, 'template<int t_bits, int MOE_TILESIZE_N>',
                  helper + '\n\ntemplate<int t_bits, int MOE_TILESIZE_N>')
    kernel = once(kernel,
        '    temp_state_u += group_idx * max_tokens_per_expert * hidden_dim;',
        '    temp_state_u += group_idx * max_tokens_per_expert * hidden_dim;\n'
        '    output_state = reinterpret_cast<float*>(temp_state_u);')
    kernel = once(kernel, '                had_hf_r_128_d_inner\n',
                  '                glm53_private_had_d\n')
    return kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', default='8,4')
    parser.add_argument('--shared-input', action='store_true')
    parser.add_argument('--tile-k', type=int, choices=(16, 32), default=32)
    parser.add_argument('--frag-stages', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--shared-stages', type=int, choices=(2, 3, 4, 6, 8, 12), default=3)
    parser.add_argument('--guarded-shared-input', action='store_true')
    parser.add_argument('--private-output', action='store_true',
                        help='Probe non-atomic per-group output in unused up-input scratch')
    args = parser.parse_args()
    source = Path('/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext')
    output = args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError('Output must not modify the installed extension')
    output.mkdir(parents=True, exist_ok=True)
    assert not (args.shared_input and args.guarded_shared_input)
    if args.private_output and not args.shared_input:
        raise ValueError('--private-output requires --shared-input')
    tree = output / ('private-source' if args.private_output else
                     'guarded-source' if args.guarded_shared_input else
                     'shared-source' if args.shared_input else 'stock-source')
    shutil.copytree(source, tree, dirs_exist_ok=True)
    kernel = tree / 'quant/exl3_moe_kernel.cuh'
    original = kernel.read_text()
    if args.shared_input or args.guarded_shared_input:
        redundant = '''                had_hf_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_u + 128 * warp_idx,
                    exp_up_suh + 128 * token_off,
                    0.088388347648f
                );'''
        old_call = 'gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);'
        assert original.count(redundant) == original.count(old_call) == 1
        if args.guarded_shared_input:
            changed = original.replace(redundant,
                '                if (exp_gate_suh != exp_up_suh) {\n' + redundant + '\n                }')
            changed = changed.replace(old_call, old_call.replace('temp_state_u',
                '(exp_gate_suh == exp_up_suh ? temp_state_g : temp_state_u)'))
        else:
            changed = original.replace(redundant, '// Shared gate/up SUH: reuse gate input.')
            changed = changed.replace(old_call, old_call.replace('temp_state_u', 'temp_state_g'))
        kernel.write_text(changed)
    if args.private_output:
        kernel.write_text(private_output_source(kernel.read_text(),
            (tree / 'quant/hadamard_inner.cuh').read_text()))
    for group in map(int, args.groups.split(',')):
        assert group in (2, 4, 8)
        label = f'g{group}' + ('_shared' if args.shared_input else '')
        if args.guarded_shared_input:
            label += '_guarded'
        if args.tile_k != 32 or args.frag_stages != 3:
            label += f'_k{args.tile_k}_f{args.frag_stages}'
        if args.shared_stages != 3:
            label += f'_s{args.shared_stages}'
        if args.private_output:
            label += '_private'
        command = ['/usr/local/cuda/bin/nvcc', '-O3', '--use_fast_math', '-std=c++17',
                   '-arch=sm_121', '--shared', '-Xcompiler=-fPIC',
                   '-Xlinker=-Bsymbolic', '-Xptxas=-v',
                   f'-DGLM53_GROUP_SIZE={group}', f'-DGLM53_TILE_K={args.tile_k}',
                   f'-DGLM53_FRAG_STAGES={args.frag_stages}',
                   f'-DGLM53_SH_STAGES={args.shared_stages}', '-I', str(tree),
                   f'-DGLM53_PRIVATE_OUTPUT={int(args.private_output)}',
                   str(Path(__file__).with_name('exl3_group_probe.cu')),
                   '-o', str(output / f'{label}.so')]
        # Bind local symbols: variants have identical C++ kernel names, but
        # must never interpose across separately loaded DSOs.
        subprocess.run(command, check=True)
        print(json.dumps({'variant': label, 'command': command,
                          'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
                          'modified_sha256': hashlib.sha256(kernel.read_bytes()).hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
