"""Native curve boundaries, radius profiles and Hair Info regressions."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
package = types.ModuleType('vibrt_hair_tests')
package.__path__ = [str(HERE)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location('vibrt_hair_tests.hair_export', HERE / 'hair_export.py')
hair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hair)


class HairExportTests(unittest.TestCase):
    def test_strands_do_not_connect_and_info_uses_arc_length(self):
        points = np.array([[0, 0, 0], [0, 0, 3], [0, 4, 3], [1, 0, 0], [1, 0, 1]], np.float32)
        arrays, info = hair._pack_curves(points, [0.02, 0.01, 0, 0.01, 0.01], [0, 3, 5], 17)
        np.testing.assert_array_equal(arrays['indices'], [[0, 1, 0], [1, 2, 1], [3, 4, 3]])
        np.testing.assert_array_equal(arrays['vertices'], points)
        np.testing.assert_allclose(info[:, 0], [0, 3/7, 1, 0, 1])
        np.testing.assert_array_equal(info[:, 1], [7, 7, 7, 1, 1])
        np.testing.assert_allclose(info[:, 2], [0.04, 0.02, 0, 0.02, 0.02])
        self.assertEqual(info[0, 3], info[2, 3])
        self.assertNotEqual(info[0, 3], info[3, 3])
        np.testing.assert_array_equal(info, hair._pack_curves(points, arrays['curve_radii'], [0, 3, 5], 17)[1])

    def test_degenerate_keys_and_zero_radius_are_not_built(self):
        points = [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 2]]
        arrays, info = hair._pack_curves(points, [1, 1, 0, 0], [0, 4])
        np.testing.assert_array_equal(arrays['indices'], [[1, 2, 1]])
        for values in arrays.values():
            self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.isfinite(info).all())

    def test_particle_radius_uses_diameter_shape_and_authored_tip(self):
        settings = types.SimpleNamespace(shape=0.0, radius_scale=0.01,
                                         root_radius=1.0, tip_radius=0.1, use_close_tip=False)
        points = np.array([[0, 0, 0], [0, 0, 3], [0, 4, 3]], np.float32)
        np.testing.assert_allclose(hair._particle_radii(settings, points),
                                   [0.005, 0.005 * (4/7 * 0.9 + 0.1), 0.0005])
        settings.shape = 0.5
        expected = 0.005 * ((4/7)**2 * 0.9 + 0.1)
        self.assertAlmostEqual(hair._particle_radii(settings, points)[1], expected)
        settings.use_close_tip = True
        self.assertEqual(hair._particle_radii(settings, points)[-1], 0)

    def test_invalid_data_rejected_before_upload(self):
        for radii, offsets in [([-1, 1], [0, 2]), ([1], [0, 2]), ([1, 1], [1, 2]),
                               ([1, np.nan], [0, 2]), ([1, 1], [0, 3])]:
            with self.assertRaises(ValueError):
                hair._pack_curves([[0, 0, 0], [0, 0, 1]], radii, offsets)

    def test_transported_normals_match_scalar_recurrence(self):
        rng = np.random.default_rng(712)
        strands = [rng.normal(size=(n, 3)).astype(np.float32)
                   for n in (0, 1, 2, 257, 257, 11, 11)]
        # Parallel up/tangent invokes the float64 cross-product fallback;
        # following keys must retain that precision, independently per strand.
        strands.extend([np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0],
                                  [1, 1, 1], [2, 1, 2]], np.float32),
                        np.zeros((5, 3), np.float32)])
        points = np.concatenate(strands)
        offsets = np.r_[0, np.cumsum([len(s) for s in strands])]
        arrays, _ = hair._pack_curves(points, np.ones(len(points)), offsets)
        expected = np.zeros_like(points)
        for first, end in zip(offsets[:-1], offsets[1:]):
            if end - first < 2:
                continue
            axes = arrays['tangents'][first:end]
            up = np.array([1, 0, 0] if abs(axes[0, 0]) < 0.9 else [0, 1, 0], np.float32)
            for i, axis in enumerate(axes):
                up = up - axis * np.dot(up, axis)
                squared = np.dot(up, up)
                if squared < 1e-12:
                    up = np.cross(axis, [0, 0, 1] if abs(axis[2]) < 0.9 else [0, 1, 0])
                    squared = np.dot(up, up)
                up = up / max(float(np.sqrt(squared)), 1e-20)
                expected[first + i] = up
        # np.dot and batched einsum can round differently across NumPy builds
        # (including 2.4). Bound accumulated float32 error over 257 keys using
        # an absolute tolerance: relative error is misleading near zero.
        actual = arrays['normals']
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-6)
        nonzero = np.linalg.norm(expected, axis=1) > 0
        np.testing.assert_allclose(np.linalg.norm(actual[nonzero], axis=1),
                                   1, rtol=0, atol=3e-7)
        np.testing.assert_allclose(
            np.einsum('ij,ij->i', actual, arrays['tangents']),
            0, rtol=0, atol=4e-6)
        # Fallback strands replay the scalar path and must remain bit-exact,
        # including float64 promotion and the all-zero degenerate strand.
        np.testing.assert_array_equal(actual[offsets[-3]:], expected[offsets[-3]:])

    def test_authored_normals_are_preserved(self):
        points = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]], np.float32)
        normals = np.tile([0, 1, 0], (4, 1)).astype(np.float32)
        arrays, _ = hair._pack_curves(points, np.ones(4), [0, 2, 4], normals=normals)
        np.testing.assert_array_equal(arrays['normals'], normals)


if __name__ == '__main__':
    unittest.main()
