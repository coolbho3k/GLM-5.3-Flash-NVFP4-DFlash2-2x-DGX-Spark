#!/usr/bin/env python3
"""CPU source contracts for the isolated private-output experiment."""
import unittest

from build_exl3_group_probe import private_output_source


class PrivateOutputTests(unittest.TestCase):
    def fixture(self):
        kernel = '''template<int t_bits, int MOE_TILESIZE_N>
    temp_state_u += group_idx * max_tokens_per_expert * hidden_dim;
gemm_up(temp_state_g, temp_intermediate_u, exp_up_trellis, K_up);
                had_hf_r_128_d_inner
'''
        helper = 'inline __device__\nvoid had_hf_r_128_d_inner\n() {\n'
        for offset in (0, 32, 64, 96):
            helper += f'    atomicAdd(output_ptr + {offset:2d} + t, sh[{offset:2d} + t]);\n'
        helper += '}\n'
        return kernel, helper

    def test_private_output_and_nonatomic_writes(self):
        kernel, helper = self.fixture()
        changed = private_output_source(kernel, helper)
        self.assertNotIn('atomicAdd', changed)
        self.assertEqual(changed.count('glm53_private_had_d'), 2)
        self.assertIn('output_state = reinterpret_cast<float*>(temp_state_u);', changed)
        self.assertEqual(changed.count('] += sh['), 4)

    def test_requires_shared_transform(self):
        kernel, helper = self.fixture()
        with self.assertRaises(ValueError):
            private_output_source(kernel.replace('gemm_up(temp_state_g',
                                                 'gemm_up(temp_state_u'), helper)

    def test_changed_upstream_anchor_rejected(self):
        kernel, helper = self.fixture()
        with self.assertRaises(ValueError):
            private_output_source(kernel, helper.replace('atomicAdd', 'changed'))


if __name__ == '__main__':
    unittest.main()
