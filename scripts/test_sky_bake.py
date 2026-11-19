"""Blender integration test: isolated sky baking preserves unsaved edits.

Run: blender -b --factory-startup --python-exit-code 1 --python scripts/test_sky_bake.py
"""

from pathlib import Path
import sys

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "blender"))
from vibrt_blender import exporter, sky_bake

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
world = bpy.data.worlds.new("Unsaved Sky")
world.use_nodes = True
scene.world = world
sky = world.node_tree.nodes.new("ShaderNodeTexSky")
sky.sky_type = "PREETHAM"
sky.sun_direction = (0.3, 0.4, 0.8)
world.node_tree.links.new(sky.outputs["Color"], world.node_tree.nodes["Background"].inputs["Color"])


def scene_ids():
    return tuple(tuple(item.as_pointer() for item in getattr(bpy.data, name))
                 for name in ("scenes", "objects", "cameras", "worlds", "materials", "images"))


previous = None
for turbidity in (2.0, 6.0):
    sky.turbidity = turbidity
    # The reference runs on this background process's main thread, without
    # an outer render. Render callbacks must always use the isolated worker.
    exporter.clear_sky_bake_cache()
    exporter.prebake_sky_envmaps_for_world(world, w=64, h=32)
    reference = dict(exporter._SKY_BAKE_CACHE)
    assert reference
    exporter.clear_sky_bake_cache()
    before = scene_ids()
    sky_bake.prepare_scene(scene, width=64, height=32)
    actual = exporter._SKY_BAKE_CACHE
    assert actual.keys() == reference.keys()
    assert scene_ids() == before, "Worker preparation mutated Blender's scene datablocks"
    assert bpy.data.filepath == "", "Worker preparation saved the user's scene"
    for key, (pixels, w, h, sun) in actual.items():
        expected, rw, rh, expected_sun = reference[key]
        assert (w, h, sun) == (rw, rh, expected_sun)
        np.testing.assert_allclose(pixels, expected, rtol=1e-5, atol=1e-5)
        if previous is not None:
            assert not np.allclose(pixels, previous), "Unsaved sky edit was ignored"
        previous = pixels.copy()
    print(f"[sky-bake-test] PASS unsaved turbidity={turbidity}", flush=True)

exporter.clear_sky_bake_cache()
sky_bake.prepare_scene(scene)
assert all((w, h) == (2048, 1024)
           for _pixels, w, h, _sun in exporter._SKY_BAKE_CACHE.values())
assert exporter._SKY_BAKE_CACHE
print("[sky-bake-test] PASS default world resolution=2048x1024", flush=True)
