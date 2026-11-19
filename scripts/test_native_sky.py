"""Compare native sky models/graphs with Cycles and verify worker fallback.

blender -b --factory-startup --python-exit-code 1 --python scripts/test_native_sky.py
No GPU is required for sky evaluation. The Cycles reference uses its CPU device.
"""

import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
from vibrt_blender import exporter, native_sky, sky_bake, vibrt_native
from vibrt_blender.bin_writer import BinWriter

os.environ.pop("VIBRT_CYCLES_SKY", None)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
world = bpy.data.worlds.new("Native sky regression")
world.use_nodes = True
scene.world = world
tree = world.node_tree
background = tree.nodes["Background"]
output = tree.nodes["World Output"]
sky = tree.nodes.new("ShaderNodeTexSky")
tree.links.new(sky.outputs["Color"], background.inputs["Color"])


def configure(model, **changes):
    sky.sky_type = model
    values = dict(sun_direction=(0.4, 0.6, 0.69), turbidity=2.9, ground_albedo=0.3,
                  sun_elevation=0.6, sun_rotation=0.8, sun_size=0.04, sun_intensity=1.0,
                  altitude=100.0, air_density=1.0, aerosol_density=1.0, ozone_density=1.0,
                  sun_disc=False)
    values.update(changes)
    for name, value in values.items():
        setattr(sky, name, value)


def ids():
    return tuple(tuple(x.as_pointer() for x in getattr(bpy.data, kind))
                 for kind in ("scenes", "objects", "worlds", "cameras", "materials", "images"))


def native_prepare(w, h):
    exporter.clear_sky_bake_cache()
    before = ids()
    with patch.object(sky_bake, "_start_worker", side_effect=AssertionError("unexpected Cycles worker")):
        sky_bake.prepare_scene(scene, width=w, height=h)
    assert ids() == before, "Native sky changed Blender datablocks"
    assert not bpy.data.filepath, "Native sky saved the scene"
    return dict(exporter._SKY_BAKE_CACHE)


def compare(actual, reference, label, tolerance=0.005):
    assert actual.shape == reference.shape and actual.dtype == np.float32
    assert np.isfinite(actual).all()
    error = float(np.abs(actual - reference).mean() / max(float(reference.mean()), 1e-6))
    # Smooth skies are compared directly, not only by average energy; this
    # catches reversed azimuth, flipped poles and a missing color transform.
    assert error < tolerance, f"{label}: relative image error {error} >= {tolerance}"
    print(f"[native-sky-test] PASS {label}: relative image error={error:.6f}", flush=True)


cases = [(model, {}) for model in ("PREETHAM", "HOSEK_WILKIE", "SINGLE_SCATTERING", "MULTIPLE_SCATTERING")]
cases += [
    ("PREETHAM", {"sun_direction": (-0.9, 0.1, 0.08), "turbidity": 4.0}),
    ("HOSEK_WILKIE", {"sun_direction": (0.8, -0.6, -0.1), "ground_albedo": 0.8}),
    ("SINGLE_SCATTERING", {"sun_elevation": 2.0, "sun_rotation": -7.0, "altitude": 0.0}),
    ("MULTIPLE_SCATTERING", {"sun_elevation": -0.1, "altitude": 5000.0, "aerosol_density": 0.3}),
]
for index, (model, changes) in enumerate(cases):
    configure(model, **changes)
    expected, _, _ = exporter._bake_sky_world_to_pixels(world, w=256, h=128)
    cache = native_prepare(256, 128)
    actual, _, _, sun = cache[exporter._sky_node_cache_key(world, sky)]
    assert sun is None
    compare(actual, expected, f"{index}: {model}")

# The solar disc needs pixel integration. Compare the complete native image
# before the renderer's separate-sun extraction, and also check irradiance.
for model in ("SINGLE_SCATTERING", "MULTIPLE_SCATTERING"):
    configure(model, sun_disc=True)
    settings = {name: getattr(sky, name) for name in (
        "turbidity", "ground_albedo", "sun_elevation", "sun_rotation", "sun_size",
        "sun_intensity", "altitude", "air_density", "aerosol_density", "ozone_density", "sun_disc",
    )}
    settings.update(model=model, sun_direction=tuple(sky.sun_direction))
    expected, _, _ = exporter._bake_sky_world_to_pixels(world, w=512, h=256)
    actual = vibrt_native.sky_pixels(settings, 512, 256)
    compare(actual, expected, model + " sun disc", tolerance=0.05)
    omega = np.sin((np.arange(256) + 0.5) * math.pi / 256)[:, None, None]
    actual_energy = (actual * omega).sum(axis=(0, 1))
    reference_energy = (expected * omega).sum(axis=(0, 1))
    np.testing.assert_allclose(actual_energy, reference_energy, rtol=0.03)

# Reflection coordinates in a world point opposite the camera ray (classroom).
configure("HOSEK_WILKIE")
coordinate = tree.nodes.new("ShaderNodeTexCoord")
tree.links.new(coordinate.outputs["Reflection"], sky.inputs["Vector"])
expected, _, _ = exporter._bake_sky_world_to_pixels(world, w=256, h=128)
actual = native_prepare(256, 128)[exporter._sky_node_cache_key(world, sky)][0]
compare(actual, expected, "world reflection coordinates")
tree.links.remove(sky.inputs["Vector"].links[0])

# HDR RGB curves, a fractional curve factor, HSV and a nonunit world strength.
configure("PREETHAM")
curves = tree.nodes.new("ShaderNodeRGBCurve")
curves.mapping.curves[3].points.new(0.4, 0.2)
curves.mapping.curves[0].points.new(0.6, 0.9)
curves.inputs[0].default_value = 0.35
hsv = tree.nodes.new("ShaderNodeHueSaturation")
hsv.inputs["Hue"].default_value = 0.6
hsv.inputs["Saturation"].default_value = 0.7
hsv.inputs["Value"].default_value = 1.4
hsv.inputs[3].default_value = 0.45
tree.links.new(sky.outputs["Color"], curves.inputs["Color"])
tree.links.new(curves.outputs["Color"], hsv.inputs["Color"])
tree.links.new(hsv.outputs["Color"], background.inputs["Color"])
background.inputs["Strength"].default_value = 2.0
expected, _, _ = exporter._bake_sky_world_to_pixels(world, w=256, h=128)
cache = native_prepare(256, 128)
compare(cache[exporter._world_generic_cache_key(world)][0], expected, "HDR curves + HSV + strength")
description = exporter._export_world(world, BinWriter(), [], [])
assert description["type"] == "envmap" and description["strength"] == 1.0

# Repeat after an unsaved edit: no stale result or disk cache is permitted.
hsv.inputs["Value"].default_value = 0.6
expected, _, _ = exporter._bake_sky_world_to_pixels(world, w=256, h=128)
changed = native_prepare(256, 128)[exporter._world_generic_cache_key(world)][0]
compare(changed, expected, "unsaved node edit")
assert not np.allclose(changed, cache[exporter._world_generic_cache_key(world)][0])

# Simple Sky -> Background applies Strength exactly once in the exporter.
tree.links.new(sky.outputs["Color"], background.inputs["Color"])
background.inputs["Strength"].default_value = 3.0
expected, _, _ = exporter._bake_sky_world_to_pixels(world, w=256, h=128)
actual = native_prepare(256, 128)[exporter._sky_node_cache_key(world, sky)][0]
description = exporter._export_world(world, BinWriter(), [], [])
compare(actual * description["strength"], expected, "direct sky strength applied once")

# A color-processed layer mixed with a constant background uses full evaluation.
tree.links.new(hsv.outputs["Color"], background.inputs["Color"])
other = tree.nodes.new("ShaderNodeBackground")
other.inputs["Color"].default_value = (0.2, 0.1, 0.3, 1.0)
mix = tree.nodes.new("ShaderNodeMixShader")
mix.inputs[0].default_value = 0.3
tree.links.new(background.outputs[0], mix.inputs[1])
tree.links.new(other.outputs[0], mix.inputs[2])
tree.links.new(mix.outputs[0], output.inputs["Surface"])
expected, _, _ = exporter._bake_sky_world_to_pixels(world, w=256, h=128)
actual = native_prepare(256, 128)[exporter._world_generic_cache_key(world)][0]
compare(actual, expected, "mixed sky and constant background")

# Camera visibility and illumination must keep their separate environments.
tree.links.new(background.outputs[0], output.inputs["Surface"])
lighting, _, _ = exporter._bake_sky_world_to_pixels(world, w=128, h=64)
tree.links.new(other.outputs[0], output.inputs["Surface"])
camera, _, _ = exporter._bake_sky_world_to_pixels(world, w=128, h=64)
path = tree.nodes.new("ShaderNodeLightPath")
tree.links.new(path.outputs["Is Camera Ray"], mix.inputs[0])
tree.links.new(mix.outputs[0], output.inputs["Surface"])
cache = native_prepare(128, 64)
compare(cache[exporter._world_branch_cache_key(world, background, "lighting")][0], lighting,
        "camera-ray split lighting")
compare(cache[exporter._world_branch_cache_key(world, other, "camera")][0], camera,
        "camera-ray split background")
assert exporter._export_world(world, BinWriter(), [], [])["split_by_camera_ray"]

# Color mixes and Gamma use the same path as curved/HSV sky graphs.
tree.links.new(background.outputs[0], output.inputs["Surface"])
background.inputs["Strength"].default_value = 1.0
color_mix = tree.nodes.new("ShaderNodeMixRGB")
color_mix.blend_type = "MULTIPLY"
color_mix.inputs[0].default_value = 0.4
color_mix.inputs[2].default_value = (0.3, 0.5, 0.8, 1.0)
tree.links.new(sky.outputs[0], color_mix.inputs[1])
gamma = tree.nodes.new("ShaderNodeGamma")
tree.links.new(color_mix.outputs[0], gamma.inputs["Color"])
tree.links.new(gamma.outputs[0], background.inputs["Color"])
for value in (1.7, 0.0):
    gamma.inputs["Gamma"].default_value = value
    expected, _, _ = exporter._bake_sky_world_to_pixels(world, w=128, h=64)
    actual = native_prepare(128, 64)[exporter._world_generic_cache_key(world)][0]
    compare(actual, expected, f"color mix + gamma={value}")
color_mix.blend_type = "DODGE"
exporter.clear_sky_bake_cache()
assert not native_sky.prepare_scene(scene, cancel=lambda: False, log=print, width=64, height=32)
assert not exporter._SKY_BAKE_CACHE, "Unsupported blend left a partial sky cache"

# Unsupported vector graphs must actually fall back, with no partial cache.
tree.links.new(background.outputs[0], output.inputs["Surface"])
tree.links.new(sky.outputs["Color"], background.inputs["Color"])
background.inputs["Strength"].default_value = 1.0
mapping = tree.nodes.new("ShaderNodeMapping")
tree.links.new(mapping.outputs[0], sky.inputs["Vector"])
exporter.clear_sky_bake_cache()
assert not native_sky.prepare_scene(scene, cancel=lambda: False, log=print, width=64, height=32)
assert not exporter._SKY_BAKE_CACHE
with patch.object(sky_bake.subprocess, "Popen", wraps=sky_bake.subprocess.Popen) as spawned:
    sky_bake.prepare_scene(scene, width=64, height=32)
    assert spawned.call_count == 1
assert exporter._SKY_BAKE_CACHE

# Reject malformed FFI inputs before entering the C++ models.
for settings, w, h in [({"model": "unknown"}, 8, 4), ({"turbidity": 0.0}, 8, 4),
                        ({"sun_direction": (0.0, 0.0, 0.0)}, 8, 4),
                        ({"sun_rotation": float("nan")}, 8, 4), ({}, 0, 4), ({}, 9000, 4)]:
    try:
        vibrt_native.sky_pixels(settings, w, h)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError(f"invalid sky settings accepted: {settings}, {w}x{h}")
print("[native-sky-test] PASS fallback, validation, unsaved edits and scene preservation", flush=True)
