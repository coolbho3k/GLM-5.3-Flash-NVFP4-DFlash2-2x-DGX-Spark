#!/usr/bin/env python3
"""Add a separately selectable weight-streaming variant of the native fast MoE.

Apply after patch_exl3_decode_pipeline.py. Preserve the validated fast kernel,
stock GEMM, and E2 prefill path byte-for-byte. This is experimental until a
serving A/B confirms the isolated screen's small kernel-time benefit.
"""
from pathlib import Path
import sys


def once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'Expected one streaming source anchor: {old!r}')
    return source.replace(old, new, 1)


def patch(root):
    root = Path(root)
    quant = root / 'quant'
    host_path = quant / 'exl3_moe.cu'
    host = host_path.read_text()
    if 'GLM53_EXL3_MOE_STREAM_WEIGHTS' in host:
        raise RuntimeError('Streaming patch already present; use an unpatched fast image')
    ptx = (root / 'ptx.cuh').read_text()
    for anchor in ('void cp_async_stream(void* smem_ptr, const void* glob_ptr)',
                   'createpolicy.fractional.L2::evict_first.b64 p, 1.0;',
                   'cp.async.cg.shared.global.L2::cache_hint'):
        if anchor not in ptx:
            raise RuntimeError(f'Upstream streaming helper changed: {anchor}')
    gemm = (quant / 'exl3_gemm_inner.cuh').read_text()
    gemm = once(gemm, 'void exl3_gemm_kernel_inner\n',
                'void glm53_exl3_stream_gemm_kernel_inner\n')
    copy = ('if (pred_b_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, '
            'gl + load_b_gl[i]);')
    gemm = once(gemm, copy, copy.replace('cp_async(', 'cp_async_stream('))
    kernel = (quant / 'glm53_exl3_moe_fast_kernel.cuh').read_text()
    kernel = once(kernel, '#include "exl3_gemm_inner.cuh"',
                  '#include "glm53_exl3_stream_gemm_inner.cuh"')
    kernel = once(kernel, 'void glm53_exl3_moe_fast_kernel(EXL3_MOE_KERNEL_ARGS)',
                  'void glm53_exl3_moe_stream_kernel(EXL3_MOE_KERNEL_ARGS)')
    # Two GEMMs, each with one fixed-bitrate and eight generic instantiations.
    if kernel.count('exl3_gemm_kernel_inner<') != 18:
        raise RuntimeError('Native MoE GEMM dispatch changed')
    kernel = kernel.replace('exl3_gemm_kernel_inner<',
                            'glm53_exl3_stream_gemm_kernel_inner<')
    wrapper = (quant / 'comp_units/glm53_exl3_moe_fast.cu').read_text()
    if wrapper.count('glm53_exl3_moe_fast') != 4:
        raise RuntimeError('Native fast wrapper changed')
    wrapper = wrapper.replace('glm53_exl3_moe_fast', 'glm53_exl3_moe_stream')
    declaration = 'fp_exl3_moe_kernel glm53_exl3_moe_fast_k4_n256(bool shared_input);'
    host = once(host, declaration, declaration + '''
fp_exl3_moe_kernel glm53_exl3_moe_stream_k4_n256(bool shared_input);

static bool glm53_stream_moe_requested() {
    static const bool requested = [] {
        const char* value = std::getenv("GLM53_EXL3_MOE_STREAM_WEIGHTS");
        TORCH_CHECK(!value || !std::strcmp(value, "0") || !std::strcmp(value, "1"),
                    "GLM53_EXL3_MOE_STREAM_WEIGHTS must be 0 or 1");
        return value && !std::strcmp(value, "1");
    }();
    return requested;
}
''')
    selection = '''        kernel = glm53_exl3_moe_fast_k4_n256(
            gate_ptrs_suh.data_ptr() == up_ptrs_suh.data_ptr());'''
    host = once(host, selection, selection + '''
        if (glm53_stream_moe_requested()) {
            kernel = glm53_exl3_moe_stream_k4_n256(
                gate_ptrs_suh.data_ptr() == up_ptrs_suh.data_ptr());
        }''')
    bindings_path = root / 'bindings.cpp'
    binding = '    m.def("glm53_fast_moe_version", []() { return 1; });'
    bindings = once(bindings_path.read_text(), binding, binding + '\n'
                    '    m.def("glm53_stream_moe_version", []() { return 1; });')
    # All validation precedes writes; existing numerical paths remain intact.
    (quant / 'glm53_exl3_stream_gemm_inner.cuh').write_text(gemm)
    (quant / 'glm53_exl3_moe_stream_kernel.cuh').write_text(kernel)
    (quant / 'comp_units/glm53_exl3_moe_stream.cu').write_text(wrapper)
    host_path.write_text(host)
    bindings_path.write_text(bindings)
    print(f'Installed opt-in native weight-streaming experiment into {root}')


if __name__ == '__main__':
    patch(sys.argv[1])
