"""Compare legacy Hair, Toon and SSS with Cycles using actual Blender nodes.

blender -b --factory-startup --python-exit-code 1 --python scripts/test_blender_reuse.py
Outputs small linear EXR/NumPy comparisons in target/blender-reuse-render.
"""
import json
import os
from pathlib import Path
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
scene.render.resolution_x, scene.render.resolution_y = 96, 64
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.cycles.device = "CPU"
scene.cycles.samples = spp = int(os.environ.get("VIBRT_REUSE_TEST_SPP", "1024"))
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.cycles.max_bounces = 8
scene.cycles.pixel_filter_type = "BOX"
scene.render.filter_size = 1
scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0
scene.vibrt_clamp_indirect = 0
world = bpy.data.worlds.new("Constant white radiance")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1
scene.world = world
bpy.ops.object.camera_add(location=(0, 0, 4))
scene.camera = bpy.context.object
scene.camera.data.type = "ORTHO"
scene.camera.data.ortho_scale = 3
cases = [("hair-reflection", "ShaderNodeBsdfHair", "Reflection"),
         ("hair-transmission", "ShaderNodeBsdfHair", "Transmission"),
         ("toon-diffuse", "ShaderNodeBsdfToon", "DIFFUSE"),
         ("toon-glossy", "ShaderNodeBsdfToon", "GLOSSY"),
         ("sss-isotropic", "ShaderNodeSubsurfaceScattering", 0.0),
         ("sss-anisotropic", "ShaderNodeSubsurfaceScattering", 0.7)]
for index, (name, kind, mode) in enumerate(cases):
    position = (index % 3 - 1, index // 3 - 0.5, 0)
    if kind == "ShaderNodeSubsurfaceScattering":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.45, location=position)
    else:
        bpy.ops.mesh.primitive_plane_add(size=0.9, location=position)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    shader = nodes.new(kind)
    shader.inputs["Color"].default_value = (0.55, 0.3, 0.15, 1)
    if kind == "ShaderNodeBsdfHair":
        shader.component = mode
        shader.inputs["RoughnessU"].default_value = 0.3
        shader.inputs["RoughnessV"].default_value = 0.4
        tangent = nodes.new("ShaderNodeCombineXYZ")
        tangent.inputs["Y"].default_value = 1
        mat.node_tree.links.new(tangent.outputs[0], shader.inputs["Tangent"])
    elif kind == "ShaderNodeBsdfToon":
        shader.component = mode
        shader.inputs["Size"].default_value = 0.4
        shader.inputs["Smooth"].default_value = 0.15
    else:
        shader.falloff = "RANDOM_WALK"
        shader.inputs["Scale"].default_value = 0.15
        shader.inputs["Radius"].default_value = (1, 0.5, 0.25)
        shader.inputs["Anisotropy"].default_value = mode
    out = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(shader.outputs[0], out.inputs["Surface"])
    bpy.context.object.data.materials.append(mat)

directory = ROOT / "target/blender-reuse-render"
directory.mkdir(parents=True, exist_ok=True)
scene.render.engine = "CYCLES"
scene.render.filepath = str(directory / "cycles.exr")
bpy.ops.render.render(write_still=True)
image = bpy.data.images.load(scene.render.filepath, check_existing=False)
reference = np.empty(96 * 64 * 4, np.float32)
image.pixels.foreach_get(reference)
reference = reference.reshape(64, 96, 4)
scene.render.engine = "VIBRT"
scene.vibrt_spp = spp
description, blobs, textures = exporter.export_scene_to_memory(bpy.context.evaluated_depsgraph_get(), serialize=False)
(directory / "scene.json").write_text(json.dumps(description, indent=2), encoding="utf-8")
actual, _ = native.render(description, [a.view(np.uint8) for a in blobs],
                          {"spp": spp, "denoise": False}, texture_arrays=textures)
assert np.isfinite(actual).all()
np.save(directory / "cuda.npy", actual)
results = []
for index, (name, _, _) in enumerate(cases):
    x, y = index % 3 * 32, index // 3 * 32
    region = (slice(y+10, y+22), slice(x+10, x+22), slice(0, 3))
    expected = reference[region].mean(axis=(0, 1))
    got = actual[region].mean(axis=(0, 1))
    error = float(np.max(np.abs(got-expected)/np.maximum(expected, 0.05)))
    results.append(dict(name=name, cycles=expected.tolist(), vibrt=got.tolist(), error=error))
    print("[blender-reuse]", results[-1], flush=True)
(directory / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
for name, pixels in (("cycles", reference), ("vibrt", actual)):
    image = bpy.data.images.new(name, width=96, height=64, alpha=True, float_buffer=True)
    image.pixels.foreach_set(pixels.ravel())
    scene.render.image_settings.file_format = "PNG"
    image.save_render(str(directory / f"{name}.png"), scene=scene)
    bpy.data.images.remove(image)
assert max(r["error"] for r in results) < 0.08, results
