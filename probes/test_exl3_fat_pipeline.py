"""Structural guards only; direct/scatter numerical gates require a GPU."""
import unittest
from pathlib import Path
import importlib.util
import tempfile
import ast
from types import SimpleNamespace
import json
import os
from build_exl3_fat_pipeline import pipelined_source, standalone, small_tile_source


class FatPipelineTests(unittest.TestCase):
    source = (Path(__file__).resolve().parents[1] / 'vendor/miaai-exl3/exl3_fat_gemm.cu').read_text()

    def test_math_and_epilogue_unchanged(self):
        result = pipelined_source(self.source)
        start = '        FragB frag_b0;'
        end = '\nvoid check_common('
        self.assertEqual(self.source[self.source.index(start):self.source.index(end)],
                         result[result.index(start):result.index(end)])

    def test_double_buffers_and_no_unconditional_tail_read(self):
        result = pipelined_source(self.source)
        self.assertIn('ring_a + 2 * FAT_TILE_M * FAT_TILE_K', result)
        self.assertIn('ring_b + 2 * FAT_N_BLOCKS * FAT_PACKED_WORDS', result)
        self.assertIn('if (m_base + a_row < size_m)', result)
        self.assertIn('*a_dst = int4{}', result)
        self.assertIn('if (k_block + 1 < size_k / FAT_TILE_K)', result)
        self.assertEqual(result.count('cp_async_wait<0>()'), 1)

    def test_standalone_has_no_torch_dependency(self):
        for source, stages in [(self.source, 1), (pipelined_source(self.source), 2)]:
            result = standalone(source, stages)
            self.assertNotIn('at::Tensor', result)
            self.assertIn(f'SHARED_BYTES = {stages} *', result)
            self.assertIn('n % 128', result)

    def test_reapply_and_drift_rejected(self):
        for source in ['', self.source + self.source, pipelined_source(self.source)]:
            with self.assertRaises(ValueError):
                pipelined_source(source)

    def test_small_tile_guards_all_a_writes(self):
        result = small_tile_source(pipelined_source(self.source))
        self.assertIn('FAT_TILE_M = 64;', result)
        self.assertLess(result.index('if (a_row < FAT_TILE_M)'), result.index('int4* a_dst'))
        self.assertIn('(m + FAT_TILE_M - 1) / FAT_TILE_M', standalone(result, 2))
        with self.assertRaises(ValueError):
            small_tile_source(self.source)

    def test_additive_native_patch_preserves_control(self):
        recipe = Path(__file__).resolve().parents[1]
        script = recipe / 'vendor/miaai-exl3/patch_exl3_fat_pipeline.py'
        spec = importlib.util.spec_from_file_location('fat_native_patch', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quant = root / 'quant'
            quant.mkdir()
            (quant / 'exl3_fat_gemm.cu').write_text(self.source)
            header = (recipe / 'vendor/miaai-exl3/exl3_fat_gemm.cuh').read_text()
            (quant / 'exl3_fat_gemm.cuh').write_text(header)
            (root / 'bindings.cpp').write_text('''#include "quant/exl3_fat_gemm.cuh"
    m.def("exl3_fat_gemm_scatter", &exl3_fat_gemm_scatter, "exl3_fat_gemm_scatter");''')
            module.patch(root)
            self.assertEqual((quant / 'exl3_fat_gemm.cu').read_text(), self.source)
            self.assertEqual((quant / 'exl3_fat_gemm.cuh').read_text(), header)
            candidate = (quant / 'exl3_fat_gemm_async2.cu').read_text()
            self.assertIn('void exl3_fat_gemm_async2(', candidate)
            self.assertIn('void exl3_fat_gemm_scatter_async2(', candidate)
            self.assertIn('#include "exl3_fat_gemm_async2.cuh"', candidate)
            m64 = (quant / 'exl3_fat_gemm_async2_m64.cu').read_text()
            self.assertIn('FAT_TILE_M = 64;', m64)
            self.assertIn('void exl3_fat_gemm_scatter_async2_m64(', m64)
            with self.assertRaises(ValueError):
                module.patch(root)

    def test_runtime_selector_is_explicit_and_fail_closed(self):
        source = Path(__file__).resolve().parents[1] / 'vendor/miaai-exl3/exl3.py'
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'select_fat_pipeline_ops')
        env = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     str(source), 'exec'), env)
        select = env['select_fat_pipeline_ops']
        old = SimpleNamespace(exl3_fat_gemm=object(), exl3_fat_gemm_scatter=object())
        self.assertEqual(select(old, 'off'), (old.exl3_fat_gemm, old.exl3_fat_gemm_scatter))
        for mode in ['m128', 'm64', 'invalid']:
            with self.assertRaises(RuntimeError):
                select(old, mode)
        old.glm53_fat_pipeline_version = lambda: 1
        with self.assertRaisesRegex(RuntimeError, 'Unsupported'):
            select(old, 'm64')
        old.glm53_fat_pipeline_version = lambda: 2
        for mode, suffix in [('m128', '_async2'), ('m64', '_async2_m64')]:
            direct, scatter = object(), object()
            setattr(old, 'exl3_fat_gemm' + suffix, direct)
            setattr(old, 'exl3_fat_gemm_scatter' + suffix, scatter)
            self.assertEqual(select(old, mode), (direct, scatter))

    def test_hot_control_retains_valid_mode_and_throttles_reads(self):
        source = Path(__file__).resolve().parents[1] / 'vendor/miaai-exl3/exl3.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'fat_pipeline_mode')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'control.json'
            clock = [1.0]
            env = {'Path': Path, 'json': json, 'os': os,
                   '_FAT_PIPELINE_CONTROL': str(path), '_FAT_PIPELINE_STATE': [0.0, 'off', None],
                   'time': SimpleNamespace(monotonic=lambda: clock[0]),
                   'logger': SimpleNamespace(info=lambda *args: None, warning=lambda *args: None)}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                         str(source), 'exec'), env)
            mode = env['fat_pipeline_mode']
            path.write_text('{"mode":"m64"}')
            self.assertEqual(mode(), 'm64')
            path.write_text('{"mode":"off"}')
            self.assertEqual(mode(), 'm64')
            clock[0] = 3.0
            path.write_text('{"mode":"broken"}')
            self.assertEqual(mode(), 'm64')
            clock[0] = 5.0
            path.write_text('{"mode":"off"}')
            self.assertEqual(mode(), 'off')


if __name__ == '__main__':
    unittest.main()
