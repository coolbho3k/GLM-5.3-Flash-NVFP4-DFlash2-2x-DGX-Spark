#!/usr/bin/env python3
"""CPU contracts for the additive native streaming-weight patch."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from test_exl3_decode_pipeline import NativePatchTests, patcher as fast_patcher
from test_exl3_weight_cache import WeightCacheTests

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stream_patch',
    ROOT / 'vendor/miaai-exl3/patch_exl3_stream_weights.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


class StreamPatchTests(unittest.TestCase):
    def fixture(self, root):
        NativePatchTests().fixture(root)
        kernel_path = root / 'quant/exl3_moe_kernel.cuh'
        kernel_path.write_text('#include "exl3_gemm_inner.cuh"\n'
            + kernel_path.read_text() + '\n' + 'exl3_gemm_kernel_inner<4>();\n' * 18)
        (root / 'quant/exl3_gemm_inner.cuh').write_text(
            'void exl3_gemm_kernel_inner\n() {\n' + WeightCacheTests.gemm + '\n}')
        (root / 'ptx.cuh').write_text(WeightCacheTests.ptx)
        fast_patcher.patch(root)

    def test_preserves_existing_math_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            preserved = ('quant/exl3_gemm_inner.cuh', 'quant/exl3_moe_kernel.cuh',
                         'quant/glm53_exl3_moe_fast_kernel.cuh',
                         'quant/comp_units/glm53_exl3_moe_fast.cu', 'ptx.cuh')
            original = {p: (root / p).read_bytes() for p in preserved}
            patcher.patch(root)
            for path, data in original.items():
                self.assertEqual((root / path).read_bytes(), data)
            stream_gemm = (root / 'quant/glm53_exl3_stream_gemm_inner.cuh').read_text()
            self.assertEqual(stream_gemm.replace('glm53_exl3_stream_gemm_kernel_inner',
                'exl3_gemm_kernel_inner').replace('cp_async_stream(', 'cp_async('),
                original['quant/exl3_gemm_inner.cuh'].decode())
            host = (root / 'quant/exl3_moe.cu').read_text()
            self.assertIn('if (glm53_stream_moe_requested())', host)
            self.assertIn('K == 4 && N_off == 1 && glm53_fast_moe_enabled(device)', host)
            self.assertIn('return value && !std::strcmp(value, "1")', host)

    def test_repeated_patch_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            patcher.patch(root)
            host = (root / 'quant/exl3_moe.cu').read_bytes()
            with self.assertRaises(RuntimeError):
                patcher.patch(root)
            self.assertEqual((root / 'quant/exl3_moe.cu').read_bytes(), host)

    def test_anchor_drift_fails_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / 'bindings.cpp').write_text('unknown binding')
            host = (root / 'quant/exl3_moe.cu').read_bytes()
            with self.assertRaises(RuntimeError):
                patcher.patch(root)
            self.assertEqual((root / 'quant/exl3_moe.cu').read_bytes(), host)
            self.assertFalse((root / 'quant/glm53_exl3_stream_gemm_inner.cuh').exists())


if __name__ == '__main__':
    unittest.main()
