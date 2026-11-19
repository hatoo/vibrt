"""RenderResult lifetime and pass updates; no Blender/GPU required."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("vibrt_progressive", Path(__file__).with_name("progressive.py"))
progressive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progressive)


class Rect:
    def foreach_set(self, pixels):
        self.pixels = np.asarray(pixels).copy()


class Engine:
    def __init__(self, names=("Combined", "Noisy Image", "Z", "Mist")):
        self.passes = {name: SimpleNamespace(name=name, rect=Rect(), channels=4 if name in
                       {"Combined", "Noisy Image"} else 1) for name in names}
        self.result = SimpleNamespace(layers=[SimpleNamespace(passes=list(self.passes.values()))])
        self.begins = []
        self.updates = []
        self.ends = []

    def begin_result(self, *args):
        self.begins.append(args)
        return self.result

    def update_result(self, result):
        self.updates.append(result)

    def end_result(self, result, **kwargs):
        self.ends.append((result, kwargs))


class ProgressiveTests(unittest.TestCase):
    def test_updates_share_one_result_until_final_passes(self):
        engine = Engine()
        first = np.ones((2, 3, 4), dtype=np.float32)
        final = first * 0.5
        depth = np.arange(6, dtype=np.float32).reshape(2, 3)
        with progressive.RenderResultDisplay(engine, 3, 2) as display:
            display.update(first)
            display.update(final)
            self.assertEqual(len(engine.ends), 0)
            np.testing.assert_array_equal(engine.passes["Combined"].rect.pixels, final.ravel())
            display.finish(final, depth, lambda d: d / 10)
        self.assertEqual(engine.begins, [(0, 0, 3, 2)])
        self.assertEqual(len(engine.updates), 3)
        self.assertEqual(engine.ends, [(engine.result, {"cancel": False, "do_merge_results": True})])
        np.testing.assert_array_equal(engine.passes["Z"].rect.pixels, depth.ravel())
        np.testing.assert_array_equal(engine.passes["Mist"].rect.pixels, (depth / 10).ravel())

    def test_cancel_preserves_last_frame_and_releases_result(self):
        engine = Engine()
        pixels = np.ones((2, 3, 4), dtype=np.float32)
        with self.assertRaises(KeyboardInterrupt):
            with progressive.RenderResultDisplay(engine, 3, 2) as display:
                display.update(pixels)
                raise KeyboardInterrupt()
        self.assertEqual(engine.ends[0][1], {"cancel": True, "do_merge_results": True})
        np.testing.assert_array_equal(engine.passes["Combined"].rect.pixels, pixels.ravel())

    def test_invalid_final_frame_releases_existing_result(self):
        engine = Engine()
        with self.assertRaisesRegex(ValueError, "depth shape"):
            with progressive.RenderResultDisplay(engine, 3, 2) as display:
                display.update(np.ones((2, 3, 4), dtype=np.float32))
                display.finish(np.ones((2, 3, 4)), np.ones((1, 1)), lambda d: d)
        self.assertTrue(engine.ends[0][1]["cancel"])

    def test_failure_before_first_preview_does_not_create_result(self):
        engine = Engine()
        with self.assertRaises(RuntimeError):
            with progressive.RenderResultDisplay(engine, 3, 2):
                raise RuntimeError("GPU setup failed")
        self.assertEqual(engine.begins, [])
        self.assertEqual(engine.ends, [])

    def test_missing_combined_pass_releases_result(self):
        engine = Engine(names=("Z",))
        with self.assertRaisesRegex(RuntimeError, "Combined"):
            with progressive.RenderResultDisplay(engine, 3, 2) as display:
                display.update(np.ones((2, 3, 4)))
        self.assertEqual(len(engine.ends), 1)

    def test_optional_passes_are_not_required(self):
        engine = Engine(names=("Combined",))
        with progressive.RenderResultDisplay(engine, 3, 2) as display:
            display.finish(np.ones((2, 3, 4)), np.ones((2, 3)), lambda d: self.fail("unused Mist"))
        self.assertEqual(len(engine.ends), 1)


if __name__ == "__main__":
    unittest.main()
