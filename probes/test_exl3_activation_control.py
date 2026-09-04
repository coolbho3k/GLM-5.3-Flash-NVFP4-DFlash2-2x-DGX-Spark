#!/usr/bin/env python3
"""Test the exact hot-control function without importing the GPU stack."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class ControlTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'vendor/miaai-exl3/exl3.py'
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'fat_activation_enabled')
        self.clock = Mock(return_value=10.0)
        self.env = {'_FUSED_FAT_ACTIVATION': False,
                    '_FAT_ACTIVATION_CONTROL': '/control.json',
                    '_FAT_ACTIVATION_STATE': [0.0, False, None],
                    'time': SimpleNamespace(monotonic=self.clock),
                    'json': json, 'Path': Path, 'logger': Mock()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), self.env)
        self.run_control = self.env['fat_activation_enabled']

    def test_disabled_path_never_reads_disk(self):
        self.env['_FAT_ACTIVATION_CONTROL'] = ''
        with patch.object(Path, 'read_text') as read:
            self.assertFalse(self.run_control())
            read.assert_not_called()

    def test_reload_is_rate_limited_and_reversible(self):
        with patch.object(Path, 'read_text', return_value='{"enabled": true}') as read:
            self.assertTrue(self.run_control())
            self.assertTrue(self.run_control())
            self.assertEqual(read.call_count, 1)
        self.clock.return_value = 12.0
        with patch.object(Path, 'read_text', return_value='{"enabled": false}'):
            self.assertFalse(self.run_control())

    def test_invalid_or_missing_file_keeps_last_good_setting(self):
        self.env['_FAT_ACTIVATION_STATE'][1] = True
        for idx, data in enumerate(('partial', '{}', '{"enabled": "false"}', '[]')):
            self.clock.return_value = 12.0 + idx * 2
            with patch.object(Path, 'read_text', return_value=data):
                self.assertTrue(self.run_control())
        self.clock.return_value = 24.0
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError()):
            self.assertTrue(self.run_control())


if __name__ == '__main__':
    unittest.main()
