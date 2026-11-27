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


if __name__ == '__main__':
    unittest.main()
