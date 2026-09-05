"""CPU-only correctness gates for EXL3 reuse accounting."""
import unittest
from analyze_exl3_weight_reuse import account, enumerate_weight_copies


class WeightReuseTests(unittest.TestCase):
    def test_c1_c2_no_redundant_row_tiles(self):
        for rows in (1, 8, 16):
            for count in range(rows + 1):
                r = account([count] * 8 + [0] * 280, rows)
                self.assertEqual(r['extra_weight_passes'], 0)

    def test_row_tile_boundaries(self):
        for count, passes in ((0, 0), (1, 1), (16, 1), (17, 2), (32, 2), (33, 3), (48, 3)):
            r = account([count], 48)
            self.assertEqual(r['m16_weight_passes'], passes)
        r = account([48] * 8, 48)
        self.assertAlmostEqual(r['ideal_copy_byte_reduction_pct'], 200 / 3)
        self.assertEqual(r['hypothetical_tiles']['32']['weight_passes'], 16)
        self.assertEqual(r['hypothetical_tiles']['64']['weight_passes'], 8)

    def test_bytes_include_three_distinct_projections(self):
        r = account([8], 8)
        self.assertEqual(r['unique_packed_weight_bytes'], 6 * 1024 * 1024)
        self.assertEqual(r['issued_packed_weight_bytes'], r['unique_packed_weight_bytes'])

    def test_copies_disjoint_even_with_split_k_and_empty_slices(self):
        for k, n in ((32, 256), (64, 512), (4096, 1024), (1024, 4096)):
            for groups in (1, 4, 8):
                r = enumerate_weight_copies(k, n, groups)
                self.assertEqual(r['duplicate_bytes'], 0)
                self.assertEqual(r['max_copies_per_address'], 1)
                self.assertEqual(r['unique_bytes'], k * n // 2)

    def test_invalid_counts_and_fat_shapes_rejected(self):
        for counts, rows in (([9], 8), ([-1], 8), ([1.5], 8), ([True], 8), ([1], 129)):
            with self.assertRaises(ValueError):
                account(counts, rows)
        for k, n in ((16, 256), (32, 128), (0, 256)):
            with self.assertRaises(ValueError):
                enumerate_weight_copies(k, n)


if __name__ == '__main__':
    unittest.main()
