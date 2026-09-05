"""CPU source guards; GPU numerical and performance tests remain mandatory."""
from pathlib import Path
import unittest
import ast
import importlib.util
import tempfile
from types import SimpleNamespace

from build_exl3_fat_pipeline import pipelined_source, small_tile_source
from build_exl3_fat_norepack import separate_gate_up_source, generated_source


class NoRepackTests(unittest.TestCase):
    original = (Path(__file__).resolve().parents[1] / 'vendor/miaai-exl3/exl3_fat_gemm.cu').read_text()

    def test_math_and_output_addressing_are_preserved(self):
        original = small_tile_source(pipelined_source(self.original))
        candidate = separate_gate_up_source(original)
        start = '        FragB frag_b0;'
        end = '\nvoid check_common('
        old = original[original.index(start):original.index(end)]
        new = candidate[candidate.index(start):candidate.index(end)]
        self.assertEqual(old.replace('svh + n_base', 'svh + packed_n_base'), new)

    def test_tile_local_selection_is_outside_k_loop(self):
        candidate = separate_gate_up_source(small_tile_source(pipelined_source(self.original)))
        self.assertLess(candidate.index('const bool is_up'), candidate.index('auto prefetch = '))
        self.assertIn('int tiles_n = half_n / 16;', candidate)
        self.assertIn('k_block * tiles_n + packed_n_base / 16', candidate)

    def test_probe_rejects_cross_boundary_tiles_and_scatter(self):
        candidate = generated_source(self.original, True)
        self.assertIn('n % 256 || scatter != 0', candidate)
        self.assertNotIn('at::Tensor', candidate)

    def test_wrong_base_and_reapplication_fail(self):
        with self.assertRaises(ValueError):
            separate_gate_up_source(self.original)
        once = separate_gate_up_source(small_tile_source(pipelined_source(self.original)))
        with self.assertRaises(ValueError):
            separate_gate_up_source(once)

    def test_native_patch_is_additive_and_checks_both_halves(self):
        path = Path(__file__).resolve().parents[1] / 'vendor/miaai-exl3/patch_exl3_fat_norepack.py'
        spec = importlib.util.spec_from_file_location('norepack_native', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'quant').mkdir()
            original = root / 'quant/exl3_fat_gemm.cu'
            original.write_text(self.original)
            (root / 'bindings.cpp').write_text('''#include "quant/exl3_fat_gemm_async2_m64.cuh"
    m.def("glm53_fat_pipeline_version", []() { return 2; });''')
            module.patch(root)
            self.assertEqual(original.read_text(), self.original)
            result = (root / 'quant/exl3_fat_norepack.cu').read_text()
            self.assertIn('check_common(a, gate, out, sg, K, mcg, mul1);', result)
            self.assertIn('check_common(a, up, out, su, K, mcg, mul1);', result)
            self.assertIn('out.size(1) == 2 * sg.numel()', result)
            self.assertIn('a.size(0) <= 8192', result)
            with self.assertRaises(ValueError):
                module.patch(root)

    def test_selector_requires_new_symbol_and_preserves_down(self):
        path = Path(__file__).resolve().parents[1] / 'vendor/miaai-exl3/exl3.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'select_fat_pipeline_ops')
        env = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        select = env['select_fat_pipeline_ops']
        ext = SimpleNamespace(glm53_fat_pipeline_version=lambda: 2)
        with self.assertRaises(RuntimeError):
            select(ext, 'm64_norepack')
        ext.glm53_fat_norepack_version = lambda: 1
        ext.exl3_fat_gate_up_norepack = object()
        ext.exl3_fat_gemm_scatter_async2_m64 = object()
        self.assertEqual(select(ext, 'm64_norepack'),
                         (ext.exl3_fat_gate_up_norepack, ext.exl3_fat_gemm_scatter_async2_m64))

    def test_all_four_weight_copies_are_skipped_together(self):
        path = Path(__file__).resolve().parents[1] / 'vendor/miaai-exl3/exl3.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'apply_exl3_batched_fat')
        branch = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == 'not no_repack')
        copies = [n for n in ast.walk(branch) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == 'copy_']
        self.assertEqual(len(copies), 4)


if __name__ == '__main__':
    unittest.main()
