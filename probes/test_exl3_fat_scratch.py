#!/usr/bin/env python3
"""CPU allocation-contract tests for the exact E2 scratch planner."""
import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class Tensor:
    def __init__(self, shape, *, dtype, device):
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.dtype = dtype

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self.dtype


class ScratchTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'vendor/miaai-exl3/exl3.py'
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_fat_scratch')
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        tree = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
        self.env = {'torch': SimpleNamespace(empty=Tensor, int16=2, float16=2, float32=4),
                    'os': os, '_FAT_SCRATCH_CACHE': {}, '_FAT_SCRATCH_BYTES': {},
                    '_EXL3_FAT_DIAG': {'fat_scratch_allocs': 0, 'fat_scratch_bytes': 0,
                                      'fat_scratch_peak_bytes': 0}}
        exec(compile(tree, str(source), 'exec'), self.env)
        self.scratch = self.env['_fat_scratch']
        self.gate = SimpleNamespace(in_features=4096, out_features=1024, K=4,
                                    trellis=SimpleNamespace(shape=(256, 64, 64)))

    def test_direct_kernel_omits_only_unused_buffers(self):
        with patch.dict(os.environ, {'EXL3_FAT_SCRATCH_ROWS': '8192'}):
            direct = self.scratch('cuda:0', 145, self.gate, use_kernel=True)
            legacy = self.scratch('cuda:0', 145, self.gate, use_kernel=False)
        self.assertEqual(set(legacy) - set(direct), {'w13', 'w2', 'down'})
        for key in direct:
            self.assertEqual(direct[key].shape, legacy[key].shape)
        size = lambda tensors: sum(t.numel() * t.element_size() for t in tensors.values())
        self.assertEqual(size(legacy) - size(direct), 152 * 1024 * 1024)

    def test_modes_do_not_alias_and_each_mode_reuses_its_cache(self):
        with patch.dict(os.environ, {'EXL3_FAT_SCRATCH_ROWS': '8192'}):
            direct = self.scratch('cuda:0', 145, self.gate, use_kernel=True)
            self.assertIs(direct, self.scratch('cuda:0', 512, self.gate, use_kernel=True))
            legacy = self.scratch('cuda:0', 145, self.gate, use_kernel=False)
        self.assertIsNot(direct, legacy)
        self.assertEqual(self.env['_EXL3_FAT_DIAG']['fat_scratch_allocs'], 2)


if __name__ == '__main__':
    unittest.main()
