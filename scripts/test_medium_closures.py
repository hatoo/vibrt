"""Cycles comparison for curved Burley scattering and homogeneous volume phases."""

from pathlib import Path
import json
import math
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
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 2048
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0
scene.cycles.pixel_filter_type = "BOX"
scene.render.filter_size = 0.01
scene.render.resolution_percentage = 100
scene.render.resolution_x, scene.render.resolution_y = 144, 96
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
world = bpy.data.worlds.new("White")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
scene.world = world
cases = ["BURLEY", "HENYEY_GREENSTEIN", "FOURNIER_FORAND", "DRAINE", "RAYLEIGH", "MIE"]
for index, name in enumerate(cases):
    location = (index % 3 - 1, index // 3 - 0.5, 0)
    if index == 0:
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=32, ring_count=16, radius=0.35, location=location
        )
        for polygon in bpy.context.object.data.polygons:
            polygon.use_smooth = True
    else:
        bpy.ops.mesh.primitive_cube_add(size=0.7, location=location)
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bpy.context.object.data.materials.append(material)
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    if index == 0:
        node = tree.nodes.new("ShaderNodeSubsurfaceScattering")
        node.falloff = "BURLEY"
        node.inputs["Color"].default_value = (0.2, 0.4, 0.7, 1)
        node.inputs["Radius"].default_value = (1, 0.5, 0.25)
        node.inputs["Scale"].default_value = 0.8
        tree.links.new(node.outputs[0], output.inputs["Surface"])
    else:
        node = tree.nodes.new("ShaderNodeVolumeCoefficients")
        node.phase = name
        node.inputs["Scatter Coefficients"].default_value = (0.15, 0.12, 0.1)
        node.inputs["Absorption Coefficients"].default_value = (0.03, 0.06, 0.1)
        tree.links.new(node.outputs[0], output.inputs["Volume"])

bpy.ops.object.camera_add(location=(0, 0, 3 / (2 * math.tan(math.pi / 6))))
scene.camera = bpy.context.object
scene.camera.data.angle = math.pi / 3
directory = ROOT / "target/medium-closures"
directory.mkdir(parents=True, exist_ok=True)
scene.render.filepath = str(directory / "cycles.exr")
bpy.ops.render.render(write_still=True)
image = bpy.data.images.load(scene.render.filepath, check_existing=False)
reference = np.empty(144 * 96 * 4, np.float32)
image.pixels.foreach_get(reference)
reference = reference.reshape(96, 144, 4)
scene.render.engine = "VIBRT"
description, blobs, textures = exporter.export_scene_to_memory(
    bpy.context.evaluated_depsgraph_get(), serialize=False
)
actual, _ = native.render(
    description,
    [blob.view(np.uint8) for blob in blobs],
    {"spp": 2048},
    texture_arrays=textures,
)
assert np.isfinite(actual).all()
np.save(directory / "cuda.npy", actual)
rows = []
for index, name in enumerate(cases):
    x, y = index % 3 * 48, index // 3 * 48
    crop = (slice(y + 18, y + 30), slice(x + 18, x + 30), slice(0, 3))
    expected = reference[crop].mean((0, 1))
    rendered = actual[crop].mean((0, 1))
    error = float(abs(expected - rendered).max())
    rows.append(
        dict(name=name, cycles=expected.tolist(), vibrt=rendered.tolist(), error=error)
    )
(directory / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
failed = [row for row in rows if row["error"] > 0.06]
for row in failed:
    print(row, flush=True)
print(
    f"[medium-closures] {len(rows)} comparisons, {len(failed)} differences", flush=True
)
assert not failed
