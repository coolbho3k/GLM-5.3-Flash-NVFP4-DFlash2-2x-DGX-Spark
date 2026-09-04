#!/usr/bin/env python3
"""CPU contracts for additive native pipeline patch and safe SUH table aliasing."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_exl3_fat_scratch import Tensor

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('decode_patch', ROOT / 'vendor/miaai-exl3/patch_exl3_decode_pipeline.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


class NativePatchTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'quant/comp_units').mkdir(parents=True)
        kernel = '''template<int t_bits, int MOE_TILESIZE_N>
void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS) {
                had_hf_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_u + 128 * warp_idx,
                    exp_up_suh + 128 * token_off,
                    0.088388347648f
                );
gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);
}
'''
        (root / 'quant/exl3_moe_kernel.cuh').write_text(kernel)
        (root / 'quant/exl3_moe.cu').write_text('#include <set>\n'
            '    fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[2 * K + N_off];\n')
        (root / 'bindings.cpp').write_text('    m.def("exl3_moe", &exl3_moe, "exl3_moe");\n')
        return kernel

    def test_patch_is_additive_and_restricted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.fixture(root)
            patcher.patch(root)
            self.assertEqual((root / 'quant/exl3_moe_kernel.cuh').read_text(), original)
            fast = (root / 'quant/glm53_exl3_moe_fast_kernel.cuh').read_text()
            self.assertIn('if constexpr (!shared_input)', fast)
            self.assertIn('shared_input ? temp_state_g : temp_state_u', fast)
            host = (root / 'quant/exl3_moe.cu').read_text()
            self.assertIn('K == 4 && N_off == 1', host)
            self.assertIn('major == 12 && minor == 1', host)
            self.assertIn('gate_ptrs_suh.data_ptr() == up_ptrs_suh.data_ptr()', host)
            self.assertIn('return value && !std::strcmp(value, "1")', host)
            with self.assertRaises(RuntimeError):
                patcher.patch(root)
            self.assertEqual((root / 'quant/exl3_moe.cu').read_text(), host)

    def test_bad_anchor_does_not_partially_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / 'bindings.cpp').write_text('unknown upstream binding')
            old_host = (root / 'quant/exl3_moe.cu').read_text()
            with self.assertRaises(RuntimeError):
                patcher.patch(root)
            self.assertEqual((root / 'quant/exl3_moe.cu').read_text(), old_host)
            self.assertFalse((root / 'quant/glm53_exl3_moe_fast_kernel.cuh').exists())


class AliasTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / 'vendor/miaai-exl3/exl3.py'
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'build_exl3_fused_state')
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        tree = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
        self.env = {'torch': SimpleNamespace(empty=Tensor, int64=8, float16=2,
                         tensor=lambda values, **kw: list(values)), 'os': os,
                    'temp_rows_fused': lambda: 128, '_FUSED_TEMP_CACHE': {},
                    '_EXL3_FAT_DIAG': {'fused_temps_allocs': 0}}
        exec(compile(tree, str(source), 'exec'), self.env)
        self.build = self.env['build_exl3_fused_state']
        self.extension = SimpleNamespace(exl3_moe_max_concurrency=lambda idx: 6)
        self.layer = SimpleNamespace(w13_trellis=SimpleNamespace(device=SimpleNamespace(index=0)),
                        _exl3_hidden_size=4096, _exl3_intermediate_local=1024, _exl3_bits=4)
        self.experts = [{name: SimpleNamespace(**{attr: SimpleNamespace(data_ptr=lambda: 123)
                          for attr in ('trellis', 'suh', 'svh')})
                          for name in ('gate', 'up', 'down')}]

    def test_alias_requires_verified_flag(self):
        with patch.dict(sys.modules, {'exllamav3_ext': self.extension}), \
             patch.dict(os.environ, {'GLM53_EXL3_MOE_FAST': '0'}):
            self.build(self.layer, self.experts)
            self.assertIsNot(self.layer._exl3_ptrs['gate_suh'], self.layer._exl3_ptrs['up_suh'])
            self.layer._exl3_shared_w13_suh = True
            self.build(self.layer, self.experts)
            self.assertIs(self.layer._exl3_ptrs['gate_suh'], self.layer._exl3_ptrs['up_suh'])
            self.assertIsNot(self.layer._exl3_ptrs['gate_svh'], self.layer._exl3_ptrs['up_svh'])

    def test_fast_mode_requires_native_version(self):
        with patch.dict(sys.modules, {'exllamav3_ext': self.extension}), \
             patch.dict(os.environ, {'GLM53_EXL3_MOE_FAST': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'requires the native'):
                self.build(self.layer, self.experts)
            self.extension.glm53_fast_moe_version = lambda: 2
            with self.assertRaisesRegex(RuntimeError, 'Unsupported'):
                self.build(self.layer, self.experts)
            self.extension.glm53_fast_moe_version = lambda: 1
            self.build(self.layer, self.experts)

    def test_stream_mode_requires_fast_and_native_version(self):
        with patch.dict(sys.modules, {'exllamav3_ext': self.extension}), \
             patch.dict(os.environ, {'GLM53_EXL3_MOE_FAST': '0',
                                    'GLM53_EXL3_MOE_STREAM_WEIGHTS': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'requires GLM53_EXL3_MOE_FAST'):
                self.build(self.layer, self.experts)
            os.environ['GLM53_EXL3_MOE_FAST'] = '1'
            self.extension.glm53_fast_moe_version = lambda: 1
            with self.assertRaisesRegex(RuntimeError, 'requires the native streaming image'):
                self.build(self.layer, self.experts)
            self.extension.glm53_stream_moe_version = lambda: 2
            with self.assertRaisesRegex(RuntimeError, 'Unsupported'):
                self.build(self.layer, self.experts)
            self.extension.glm53_stream_moe_version = lambda: 1
            self.build(self.layer, self.experts)
            os.environ['GLM53_EXL3_MOE_STREAM_WEIGHTS'] = 'bad'
            with self.assertRaisesRegex(RuntimeError, 'must be 0 or 1'):
                self.build(self.layer, self.experts)


if __name__ == '__main__':
    unittest.main()
