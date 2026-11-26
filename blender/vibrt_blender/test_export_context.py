"""Exporter state isolation without Blender or a native extension."""
import importlib
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

package = types.ModuleType("vibrt_context_tests")
package.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules[package.__name__] = package


def module(name):
    return importlib.import_module(f"{package.__name__}.{name}")


ExportContext = module("export_context").ExportContext
textures = module("texture_export")
BinWriter = module("bin_writer").BinWriter
log = module("_log")
with patch.dict(sys.modules, {"bpy": types.ModuleType("bpy")}):
    materials = module("material_export")


class Image:
    name = "shared image"
    size = (4, 2)
    colorspace_settings = types.SimpleNamespace(name="Linear")

    def __init__(self):
        self.value = 0.25
        self.reads = 0
        self.pixels = self

    def foreach_get(self, output):
        self.reads += 1
        output[:] = self.value

    def as_pointer(self):
        return 42


class ExportContextTests(unittest.TestCase):
    def test_image_cache_is_owned_by_one_export(self):
        image = Image()
        first, second = ExportContext(), ExportContext()
        a = textures._image_to_linear_rgb(image, context=first)
        self.assertIs(textures._image_to_linear_rgb(image, context=first), a)
        self.assertFalse(a[0].flags.writeable)
        image.value = 0.75
        b = textures._image_to_linear_rgb(image, context=second)
        np.testing.assert_array_equal(a[0], np.full((2, 4, 3), 0.25, np.float32))
        np.testing.assert_array_equal(b[0], np.full((2, 4, 3), 0.75, np.float32))
        self.assertEqual(image.reads, 2)
        self.assertEqual(first.stats["linear_rgb_cache_hits"], 1)
        self.assertEqual(second.stats["linear_rgb_cache_hits"], 0)
        materials.reset_stats(context=first)
        self.assertTrue(second.linear_rgb_cache)
        self.assertFalse(first.linear_rgb_cache)

    def test_texture_scale_and_stats_do_not_leak(self):
        image = Image()
        half, full = ExportContext(texture_pct=50), ExportContext()
        for context, size in ((half, (2, 1)), (full, (4, 2))):
            writer, entries = BinWriter(), []
            textures.export_image_texture(image, writer, entries, context=context)
            self.assertEqual((entries[0]["width"], entries[0]["height"]), size)
            self.assertEqual(writer.texture_arrays[0].size, size[0] * size[1] * 4)
            self.assertEqual(context.stats["texture_count"], 1)
        materials.end_export(context=half)
        self.assertIsNone(full.texture_pct)
        snapshot = materials.pop_stats(context=full)
        snapshot["slow_textures"].append("local edit")
        self.assertFalse(full.stats["slow_textures"])

    def test_material_identity_restored_on_nested_export_and_error(self):
        context = ExportContext(material="caller")
        inner = types.SimpleNamespace(name="inner", use_nodes=False, diffuse_color=(1, 0, 0, 1))
        observed = []

        class Outer:
            name, use_nodes = "outer", False

            @property
            def diffuse_color(self):
                observed.append(context.material)
                materials.export_material(inner, BinWriter(), [], context=context)
                observed.append(context.material)
                raise ValueError("failed material")

        with self.assertRaisesRegex(ValueError, "failed material"):
            materials.export_material(Outer(), BinWriter(), [], context=context)
        self.assertEqual(observed, ["outer", "outer"])
        self.assertEqual(context.material, "caller")

    def test_warning_dedup_and_sky_maps_are_export_local(self):
        first, second = ExportContext(material="paint"), ExportContext(material="paint")
        messages = []
        with log.redirect(messages.append):
            first.warn("missing", "image missing")
            first.warn("missing", "image missing")
            second.warn("missing", "image missing")
        self.assertEqual(len(messages), 2)
        first.sky_maps["world"] = np.zeros((2, 4, 3), np.float32)
        first.group_stack.append("group")
        self.assertFalse(second.sky_maps)
        self.assertFalse(second.group_stack)

    def test_log_redirect_is_thread_local_and_restores_after_failure(self):
        outer, inner, child = [], [], []
        entered, release = threading.Event(), threading.Event()

        def worker():
            with log.redirect(child.append):
                entered.set()
                release.wait(5)
                log.log("child")

        with log.redirect(outer.append):
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                log.log("outer")
            finally:
                release.set()
                thread.join()
            with self.assertRaises(ValueError), log.redirect(inner.append):
                log.log("nested")
                raise ValueError("failed export")
            log.log("restored")
        self.assertEqual(outer, ["outer", "restored"])
        self.assertEqual(inner, ["nested"])
        self.assertEqual(child, ["child"])


if __name__ == "__main__":
    unittest.main()
