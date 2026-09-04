#!/usr/bin/env python3
"""CPU guards for the probe-only shared-memory/residency experiment."""
from pathlib import Path
import unittest

from build_exl3_group_probe import validate_tight_smem_layout


class OccupancyTests(unittest.TestCase):
    def fixture(self):
        return '''
const int sh_a_stage_size = TILESIZE_M * TILESIZE_K;
const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
const int sh_c_size = MAX(4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP,
    shmem_out_had ? TILESIZE_N * TILESIZE_M : 0);
const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);
float* sh_c = (float*) (sh_b + sh_b_stage_size * SH_STAGES);
'''

    def test_expected_layout(self):
        validate_tight_smem_layout(self.fixture())

    def test_larger_upstream_layout_rejected(self):
        with self.assertRaises(ValueError):
            validate_tight_smem_layout(self.fixture().replace(
                'TILESIZE_M * TILESIZE_K;', 'TILESIZE_M * TILESIZE_K * 2;'))

    def test_cooperative_launch_and_residency_gate_present(self):
        source = Path(__file__).with_name('exl3_group_probe.cu').read_text()
        self.assertIn('cudaOccupancyMaxActiveBlocksPerMultiprocessor', source)
        self.assertIn('probe_resident_limit < GLM53_RESIDENT_BLOCKS', source)
        self.assertIn('cudaDevAttrCooperativeLaunch', source)
        self.assertIn('cudaLaunchCooperativeKernel', source)
        self.assertIn('concurrency > probe_grid_groups', source)


if __name__ == '__main__':
    unittest.main()
