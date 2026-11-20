"""CPU regressions for ribbon geometry, batching and Hair Info attributes."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_package = types.ModuleType("vibrt_hair_tests")
_package.__path__ = [str(_HERE)]
sys.modules[_package.__name__] = _package
_spec = importlib.util.spec_from_file_location(
    "vibrt_hair_tests.hair_export", _HERE / "hair_export.py"
)
hair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hair)


class HairExportTests(unittest.TestCase):
    def test_bent_strand_preserves_scalar_export_geometry(self):
        # Captured from the original scalar exporter, before batching.
        points = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]], np.float32)
        edges = np.array([
            [-0.0149616115, 0.00107246370, 0],
            [0.0149616115, -0.00107246370, 0],
            [-0.00713254884, 0.000639085541, 1.00356627],
            [0.00713254884, -0.000639085541, 0.996433735],
            [0.999295413, 0.0000841770816, 2.00070453],
            [1.00070465, -0.0000841770816, 1.99929535],
        ], np.float32)
        normals = np.array([
            [-0.0756802186, -0.996972084, 0.0178656783],
            [-0.0674120411, -0.996541440, 0.0485878475],
        ], np.float32)
        axes = np.array([
            [0, 0, 1], [0.447213590, 0, 0.894427180],
            [0.707106769, 0, 0.707106769],
        ], np.float32)
        actual = hair._strands_to_ribbons(
            np.stack([points, points]), np.array([0.03, 0.03], np.float32),
            np.array([0.002, 0.002], np.float32), [17, 17],
        )
        keys = [0, 1, 0, 0, 1, 1, 1, 2, 1, 1, 2, 2]
        expected = (
            edges[[0, 2, 1, 1, 2, 3, 2, 4, 3, 3, 4, 5]],
            np.repeat(normals, 6, axis=0), axes[keys],
            np.column_stack(([0, 0, 1, 1, 0, 1] * 2, np.array(keys) / 2)),
        )
        for result, reference in zip(actual, expected):
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_allclose(
                result, np.tile(reference, (2, 1)), rtol=1e-6, atol=1e-7
            )

    def test_batches_preserve_order_seeds_and_bound_temporary_storage(self):
        # Short/degenerate strands still consume their original random index.
        sizes = [0, 1, 2, 2, 3, 2] + [3] * 260 + [1024] * 70
        strands = [(np.full((n, 3), i, np.float32), 0.01, 0.002)
                   for i, n in enumerate(sizes)]
        batches = list(hair._strand_batches(iter(strands), 123))
        seen = []
        for batch in batches:
            self.assertLessEqual(len(batch), 256)
            self.assertLessEqual(len(batch) * len(batch[0][0]), 65536)
            self.assertEqual(len({len(s[0]) for s in batch}), 1)
            for points, root, tip, seed in batch:
                index = int(points[0, 0])
                seen.append(index)
                self.assertEqual(seed, (123 ^ (index * 49157)) & 0xFFFFFFFF)
                self.assertEqual((root, tip), (0.01, 0.002))
        self.assertEqual(seen, [i for i, n in enumerate(sizes) if n >= 2])

    def test_degenerate_strands_stay_finite(self):
        points = np.zeros((2, 4, 3), np.float32)
        p, n, t, uv = hair._strands_to_ribbons(
            points, np.ones(2, np.float32), np.zeros(2, np.float32), [0, 9]
        )
        for data in (p, n, t, uv):
            self.assertTrue(np.isfinite(data).all())
        np.testing.assert_array_equal(p, 0)
        np.testing.assert_array_equal(n, np.tile([0, 0, 1], (36, 1)))
        self.assertIsNone(hair._strand_to_ribbon(points[0, :1], 1, 0, 0))

    def test_hair_info_uses_arc_length_and_preserves_random_values(self):
        points = np.array([[[0, 0, 0], [0, 0, 3], [0, 4, 3]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], np.float32)
        roots = np.array([0.04, 0.02], np.float32)
        tips = np.array([0.01, 0.02], np.float32)
        seeds = [0x12345678, 9]
        uv = hair._strands_to_ribbons(points, roots, tips, seeds)[3]
        info = hair._hair_info(points, roots, tips, seeds, uv).reshape(2, 12, 4)
        np.testing.assert_allclose(info[0, :, 0],
                                   [0, 3/7, 0, 0, 3/7, 3/7, 3/7, 1, 3/7, 3/7, 1, 1])
        np.testing.assert_array_equal(info[0, :, 1], 7)
        np.testing.assert_allclose(info[0, :, 2],
                                   [0.04, 0.025, 0.04, 0.04, 0.025, 0.025,
                                    0.025, 0.01, 0.025, 0.025, 0.01, 0.01])
        np.testing.assert_array_equal(info[0, :, 3], 0x345678 / 16777216)
        np.testing.assert_array_equal(info[1, :, :2], 0)
        self.assertTrue(np.isfinite(info).all())


if __name__ == "__main__":
    unittest.main()
