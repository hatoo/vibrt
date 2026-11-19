"""CPU binding/array regressions; requires a staged native extension, no Blender."""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

package = types.ModuleType("vibrt_array_tests")
package.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules[package.__name__] = package
ops = importlib.import_module(package.__name__ + ".node_ops")
fold = importlib.import_module(package.__name__ + ".color_fold")


@unittest.skipUnless(ops.native(), "Build/stage the native extension to test Python array bindings")
class NodeArrayTests(unittest.TestCase):
    def test_noncontiguous_input_and_snapshot(self):
        source = np.arange(120, dtype=np.float32).reshape(5, 8, 3) / 120
        view = source[::-1, ::2]
        result = ops.apply(dict(type="invert"), view)
        np.testing.assert_allclose(result, 1.0-view, atol=1e-7)
        saved = result.copy()
        source[:] = 0
        np.testing.assert_array_equal(result, saved)

    def test_constant_color_and_per_pixel_factors(self):
        other = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 18
        factor = np.linspace(0, 1, 6, dtype=np.float32).reshape(2, 3, 1)
        a = np.array([0.1, 0.3, 0.8], dtype=np.float32)
        expected = a * (1-factor) + other * factor
        actual = ops.apply(dict(type="mix"), a, other, factor)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        np.testing.assert_array_equal(a, [np.float32(0.1), np.float32(0.3), np.float32(0.8)])

    def test_factor_drives_broadcast_shape(self):
        factors = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
        result = ops.apply(dict(type="mix"), [0, 0, 0], [1, 1, 1], factors)
        np.testing.assert_array_equal(result, np.repeat(factors, 3, axis=1))

    def test_invalid_shape_and_nonfinite_data(self):
        for value in ([1, 2], [1, 2, float("nan")]):
            with self.assertRaises(ValueError):
                ops.apply(dict(type="invert"), value)


class OptionalFoldTests(unittest.TestCase):
    def test_missing_extension_keeps_graph_intact(self):
        graph = {"nodes": [{"type": "const", "rgb": [0.2, 0.3, 0.4]},
                            {"type": "invert", "input": 0}], "output": 1}
        with patch.object(ops, "native", return_value=None):
            result = fold.fold_constants(graph)
        self.assertEqual(result, graph)
        self.assertIsNot(result["nodes"], graph["nodes"])


if __name__ == "__main__":
    unittest.main()
