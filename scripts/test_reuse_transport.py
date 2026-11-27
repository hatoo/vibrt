"""Compare Cycles camera filters, aperture shapes and finite-light volume transport.

blender -b --factory-startup --python-exit-code 1 --python scripts/test_reuse_transport.py
Artifacts are kept in target/reuse-transport.
"""
from pathlib import Path
import json
import math
import os
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
import vibrt_blender
from vibrt_blender import exporter, vibrt_native as native

bpy.ops.wm.read_factory_settings(use_empty=True)
vibrt_blender.register()
scene = bpy.context.scene
size = 32
spp = int(os.environ.get("VIBRT_TRANSPORT_TEST_SPP", "2048"))
scene.render.resolution_x = scene.render.resolution_y = size
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.cycles.device = "CPU"
scene.cycles.samples = spp
scene.cycles.use_adaptive_sampling = scene.cycles.use_denoising = False
scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0
scene.vibrt_clamp_indirect = 0
scene.cycles.max_bounces = 12
scene.cycles.volume_bounces = 12
world = bpy.data.worlds.new("Black")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0
scene.world = world
bpy.ops.object.camera_add(location=(0, 0, 4))
scene.camera = bpy.context.object
camera = scene.camera.data
camera.angle = math.pi / 4
directory = ROOT / "target/reuse-transport"
directory.mkdir(parents=True, exist_ok=True)
results = []


def compare(name, tolerance=0.02):
    if os.environ.get("VIBRT_TEST_FILTER", "") not in name:
        return
    scene.render.engine = "CYCLES"
    scene.render.filepath = str(directory / f"{name}.exr")
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    pixels = np.empty(size * size * 4, np.float32)
    image.pixels.foreach_get(pixels)
    reference = pixels.reshape(size, size, 4)
    bpy.data.images.remove(image)
    scene.render.engine = "VIBRT"
    desc, blobs, textures = exporter.export_scene_to_memory(
        bpy.context.evaluated_depsgraph_get(), serialize=False)
    actual, _ = native.render(desc, [a.view(np.uint8) for a in blobs],
                              {"spp": spp, "denoise": False}, texture_arrays=textures)
    assert np.isfinite(actual).all(), name
    # Pixel error catches aperture shape/filter width errors that preserve energy.
    scale = max(float(reference[..., :3].mean()), 0.02)
    error = float(abs(actual[..., :3] - reference[..., :3]).mean()) / scale
    energy_error = float(abs(actual[..., :3].mean() - reference[..., :3].mean())) / scale
    results.append(dict(name=name, relative_mae=error, energy_error=energy_error,
                        cycles=float(reference[..., :3].mean()), vibrt=float(actual[..., :3].mean())))
    (directory / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.save(directory / f"{name}.npy", actual)
    print("[transport]", results[-1], flush=True)
    assert error < tolerance, results[-1]
    assert energy_error < tolerance / 2, results[-1]


bpy.ops.mesh.primitive_plane_add(size=2)
plane = bpy.context.object
mat = bpy.data.materials.new("Pattern")
mat.use_nodes = True
plane.data.materials.append(mat)
tree = mat.node_tree
tree.nodes.clear()
output = tree.nodes.new("ShaderNodeOutputMaterial")
emission = tree.nodes.new("ShaderNodeEmission")
texture = tree.nodes.new("ShaderNodeTexChecker")
texture.inputs["Scale"].default_value = 5
texture.inputs["Color1"].default_value = (0.1, 0.3, 0.7, 1)
texture.inputs["Color2"].default_value = (0.8, 0.2, 0.1, 1)
tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
tree.links.new(emission.outputs[0], output.inputs["Surface"])
for filter_type in ("BOX", "GAUSSIAN", "BLACKMAN_HARRIS"):
    scene.cycles.pixel_filter_type = filter_type
    scene.cycles.filter_width = 1.5
    compare(f"filter-{filter_type}")
camera.dof.use_dof = True
camera.dof.focus_distance = 0.4
camera.dof.aperture_fstop = 0.7
for blades, ratio in ((0, 1), (3, 1), (6, 1.7)):
    camera.dof.aperture_blades = blades
    camera.dof.aperture_ratio = ratio
    camera.dof.aperture_rotation = 0.4
    compare(f"aperture-{blades}-{ratio}", 0.03)
camera.type = "ORTHO"
camera.ortho_scale = 3.2
camera.dof.aperture_fstop = 2
camera.dof.focus_distance = 2
compare("aperture-orthographic", 0.03)
camera.type = "PERSP"

# A finite emitter within a bounded medium exercises the equiangular proposal
# and its mixture with distance sampling, including colored absorption.
camera.dof.use_dof = False
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1
bpy.data.objects.remove(plane, do_unlink=True)
bpy.ops.mesh.primitive_cube_add(size=2)
box = bpy.context.object
mat = bpy.data.materials.new("Medium")
mat.use_nodes = True
box.data.materials.append(mat)
tree = mat.node_tree
tree.nodes.clear()
output = tree.nodes.new("ShaderNodeOutputMaterial")
volume = tree.nodes.new("ShaderNodeVolumeCoefficients")
volume.inputs["Scatter Coefficients"].default_value = (0.25, 0.2, 0.15)
volume.inputs["Absorption Coefficients"].default_value = (0.05, 0.1, 0.2)
tree.links.new(volume.outputs[0], output.inputs["Volume"])
for kind in ("POINT", "AREA"):
    data = bpy.data.lights.new("Finite emitter", kind)
    data.energy = 10
    light = bpy.data.objects.new("Finite emitter", data)
    scene.collection.objects.link(light)
    light.location = (0.6, 0, 0.2)
    light.visible_camera = False
    if kind == "POINT":
        data.shadow_soft_size = 0
    else:
        data.size = 0.4
    compare(f"volume-{kind}", 0.08)
    bpy.data.objects.remove(light, do_unlink=True)
