#!/usr/bin/env python3
"""Regression tests for the strict four-over-six image patcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "overlay-dflash2" / "patch_b12x_nvfp4_four_over_six.py"
SPEC = importlib.util.spec_from_file_location("four_over_six_patcher", PATCHER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FourOverSixPatchTest(unittest.TestCase):
    def fixture(self) -> str:
        return (
            MODULE.DOC_OLD
            + MODULE.IMPORT_OLD
            + "prefix\n"
            + MODULE.HELPER_ANCHOR
            + "middle\n"
            + MODULE.WRITER_OLD
            + "suffix\n"
            + MODULE.SPEC_OLD
        )

    def test_patch_is_complete_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "kv_cache.py"
            target.write_text(self.fixture())
            self.assertTrue(MODULE.patch(target))
            result = target.read_text()
            self.assertIn(MODULE.MARKER, result)
            self.assertIn("fp4_decode_4bytes", result)
            self.assertIn("packed4", result)
            self.assertIn("\n        5,\n", result)
            self.assertFalse(MODULE.patch(target))
            self.assertEqual(result, target.read_text())

    def test_source_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "kv_cache.py"
            target.write_text("unexpected upstream source\n")
            with self.assertRaisesRegex(
                RuntimeError, "record-layout documentation"
            ):
                MODULE.patch(target)


if __name__ == "__main__":
    unittest.main()
