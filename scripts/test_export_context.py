"""Blender regression for independent material, image and sky export state.

blender -b --factory-startup --python-exit-code 1 --python scripts/test_export_context.py
"""
from pathlib import Path
import sys

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender"))
import vibrt_blender
from vibrt_blender import exporter, sky_bake
from vibrt_blender.export_context import ExportContext

vibrt_blender.register()
scene = bpy.context.scene
scene.render.engine = "VIBRT"
material = bpy.data.objects["Cube"].active_material
image = bpy.data.images.new("Context image", 8, 4, float_buffer=True)
image.pixels.foreach_set(np.tile([0.25, 0.5, 0.75, 1.0], 32).astype(np.float32))
tex = material.node_tree.nodes.new("ShaderNodeTexImage")
tex.image = image
material.node_tree.links.new(tex.outputs["Color"], material.node_tree.nodes["Principled BSDF"].inputs["Base Color"])

world = scene.world
sky = world.node_tree.nodes.new("ShaderNodeTexSky")
sky.sky_type = "PREETHAM"
world.node_tree.links.new(sky.outputs["Color"], world.node_tree.nodes["Background"].inputs["Color"])
small, large = ExportContext(), ExportContext()
sky_bake.prepare_scene(scene, width=32, height=16, context=small)
saved_sky = next(iter(small.sky_maps.values()))[0].copy()
sky.turbidity = 8.0
sky_bake.prepare_scene(scene, width=32, height=16, context=large)
assert not np.array_equal(saved_sky, next(iter(large.sky_maps.values()))[0])
np.testing.assert_array_equal(saved_sky, next(iter(small.sky_maps.values()))[0])


def export(context, pct):
    desc, _, arrays = exporter.export_scene_to_memory(
        bpy.context.evaluated_depsgraph_get(), serialize=False, texture_pct=pct, context=context,
    )
    texture = next(t for t in desc["textures"] if (t["width"], t["height"]) != (32, 16))
    return texture, arrays[texture["array_index"]].copy()


tiny, before = export(small, 25)
full, _ = export(large, 100)
assert (tiny["width"], tiny["height"]) == (2, 1)
assert (full["width"], full["height"]) == (8, 4)
assert small.texture_pct is None and large.texture_pct is None
assert small.material == large.material == ""
image.pixels.foreach_set(np.tile([1.0, 0.0, 0.0, 1.0], 32).astype(np.float32))
_, after = export(small, 25)
assert not np.array_equal(before, after), "Reused export context retained stale image pixels"
assert not small.group_stack and not large.group_stack
print("[export-context-test] PASS independent skies, shader textures, scales, and image edits", flush=True)
