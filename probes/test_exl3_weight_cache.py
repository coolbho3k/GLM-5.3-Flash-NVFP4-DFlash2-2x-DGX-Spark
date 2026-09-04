#!/usr/bin/env python3
"""CPU-only source guards for packed-weight cache-policy probes."""
import unittest

from build_exl3_group_probe import weight_cache_source


class WeightCacheTests(unittest.TestCase):
    gemm = '''if (pred_a_gl[i]) cp_async(sh + load_a_sh[i], gl + load_a_gl[i]);
if (pred_b_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i]);
cp_async_fence();'''
    ptx = '''__device__ inline void cp_async(void* smem_ptr, const void* glob_ptr)
{
    const int bytes = 16;
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;");
}
__device__ inline void cp_async_stream(void* smem_ptr, const void* glob_ptr)
{
    // createpolicy.fractional.L2::evict_first.b64 p, 1.0;
    // cp.async.cg.shared.global.L2::cache_hint
}
'''

    def test_default_unchanged(self):
        self.assertEqual(weight_cache_source(self.gemm, self.ptx, 'default'),
                         (self.gemm, self.ptx))

    def test_stream_only_changes_weight_call(self):
        gemm, ptx = weight_cache_source(self.gemm, self.ptx, 'stream')
        self.assertEqual(ptx, self.ptx)
        self.assertEqual(gemm.replace('cp_async_stream(', 'cp_async('), self.gemm)
        self.assertIn('if (pred_b_gl[i]) cp_async_stream(', gemm)

    def test_prefetch_preserves_original_helper_and_activation_load(self):
        gemm, ptx = weight_cache_source(self.gemm, self.ptx, 'prefetch128')
        self.assertEqual(gemm.replace('glm53_cp_async_prefetch128(', 'cp_async('), self.gemm)
        self.assertIn('cp.async.cg.shared.global.L2::128B [%0], [%1], %2;', ptx)
        self.assertIn('cp.async.cg.shared.global [%0], [%1], %2;', ptx)
        self.assertEqual(ptx.count('const int bytes = 16;'), 2)

    def test_missing_or_duplicate_weight_anchor_rejected(self):
        for gemm in ('', self.gemm + self.gemm):
            with self.assertRaises(ValueError):
                weight_cache_source(gemm, self.ptx, 'stream')

    def test_changed_helper_rejected(self):
        for policy, ptx in [('stream', self.ptx.replace('evict_first', 'evict_last')),
                            ('prefetch128', self.ptx.replace('bytes = 16', 'bytes = 8'))]:
            with self.assertRaises(ValueError):
                weight_cache_source(self.gemm, ptx, policy)


if __name__ == '__main__':
    unittest.main()
