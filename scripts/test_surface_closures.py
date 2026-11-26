"""Compare Cycles surface closures with the shared CPU/CUDA implementation.

blender -b --factory-startup --python-exit-code 1 --python scripts/test_surface_closures.py
Writes EXR/PNG comparisons and per-material measurements under target/surface-closures.
"""
import json
import math
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
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.cycles.device = "CPU"
SPP = int(os.environ.get("VIBRT_SURFACE_TEST_SPP", "8192"))
scene.cycles.samples = SPP
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.cycles.sample_clamp_direct = 0
scene.cycles.sample_clamp_indirect = 0
scene.vibrt_clamp_indirect = 0
scene.cycles.max_bounces = 4
scene.cycles.pixel_filter_type = "BOX"
# Match vibrt's pixel-wide box filter.
scene.render.filter_size = 1.0
scene.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("Uniform environment")
world.use_nodes = True
scene.world = world
cases = []


def material(name, kind, values, properties=None, linked=()):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    node = tree.nodes.new(kind)
    for prop, value in (properties or {}).items():
        setattr(node, prop, value)
    for socket, value in values.items():
        sock = node.inputs.get(socket)
        if sock is not None:
            sock.default_value = value
    for socket in linked:
        if node.inputs.get(socket) is None:
            continue
        value = values[socket]
        if isinstance(value, tuple):
            source = tree.nodes.new("ShaderNodeRGB")
            source.outputs[0].default_value = (*value[:3], 1)
        else:
            source = tree.nodes.new("ShaderNodeValue")
            source.outputs[0].default_value = value
        tree.links.new(source.outputs[0], node.inputs[socket])
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(node.outputs[0], output.inputs["Surface"])
    cases.append((name, mat))


color = (0.55, 0.27, 0.12, 1)
for roughness in (0.0, 0.35, 0.8):
    material(f"diffuse-{roughness}", "ShaderNodeBsdfDiffuse", {"Color": color, "Roughness": roughness})
    for mode in ("MICROFIBER", "ASHIKHMIN"):
        material(f"sheen-{mode}-{roughness}", "ShaderNodeBsdfSheen",
                 {"Color": color, "Roughness": roughness}, {"distribution": mode})
    for kind in ("Anisotropic", "Glass", "Refraction"):
        for distribution in ("GGX", "BECKMANN", "MULTI_GGX"):
            if kind == "Refraction" and distribution == "MULTI_GGX":
                continue
            values = {"Color": color, "Roughness": roughness}
            if kind == "Anisotropic":
                values["Anisotropy"] = 0.6
            material(f"{kind}-{distribution}-{roughness}", "ShaderNodeBsdf" + kind,
                     values, {"distribution": distribution})

for mode in ("F82", "PHYSICAL_CONDUCTOR"):
    values = {"Base Color": color, "Edge Tint": (0.3, 0.8, 0.6, 1),
              "IOR": (0.3, 0.8, 1.2), "Extinction": (3.0, 2.0, 1.0), "Roughness": 0.25}
    material("metal-" + mode, "ShaderNodeBsdfMetallic", values, {"fresnel_type": mode})
    material("metal-film-" + mode, "ShaderNodeBsdfMetallic",
             dict(values, **{"Thin Film Thickness": 420.0, "Thin Film IOR": 1.4}),
             {"fresnel_type": mode}, linked=("Thin Film Thickness", "IOR", "Extinction", "Edge Tint"))

base = {"Base Color": color, "Roughness": 0.4}
variants = {
    "base": {},
    "diffuse-rough": {"Diffuse Roughness": 0.8},
    "specular-tint": {"Specular Tint": (0.2, 0.6, 1.0, 1)},
    "specular-level": {"Specular IOR Level": 0.15},
    "metal": {"Metallic": 0.7, "Specular Tint": (0.4, 0.8, 0.9, 1)},
    "glass": {"Transmission Weight": 0.8, "IOR": 1.6},
    "coat": {"Coat Weight": 0.8, "Coat Roughness": 0.25, "Coat IOR": 1.8},
    "coat-tint": {"Coat Weight": 0.8, "Coat Tint": (0.2, 0.7, 0.9, 1)},
    "coat-normal": {"Coat Weight": 0.8, "Coat Normal": (0.25, 0.1, 0.963), "Coat Roughness": 0.3},
    "sheen": {"Sheen Weight": 0.8, "Sheen Roughness": 0.4, "Sheen Tint": (0.3, 0.7, 0.9, 1)},
    "film": {"Thin Film Thickness": 350.0, "Thin Film IOR": 1.4, "Metallic": 0.5},
    "layers": {"Coat Weight": 0.7, "Coat Tint": (0.5, 0.8, 1.0, 1), "Sheen Weight": 0.6,
               "Sheen Tint": (0.8, 0.3, 0.2, 1), "Metallic": 0.3, "Transmission Weight": 0.2},
    "emission-layers": {"Coat Weight": 0.9, "Coat Tint": (0.2, 0.7, 0.9, 1), "Sheen Weight": 0.8,
                        "Emission Color": (0.3, 0.5, 0.1, 1), "Emission Strength": 2.0},
}
for name, extra in variants.items():
    material("principled-" + name, "ShaderNodeBsdfPrincipled", dict(base, **extra), linked=tuple(extra))

side = math.ceil(math.sqrt(len(cases)))
tile = 20
resolution = side * tile
scene.render.resolution_x = scene.render.resolution_y = resolution
camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
scene.collection.objects.link(camera)
scene.camera = camera
camera.data.angle = math.pi / 2
camera.location = (0, 0, side / 2)
for i, (name, mat) in enumerate(cases):
    bpy.ops.mesh.primitive_plane_add(size=1, location=(i % side - side / 2 + 0.5, i // side - side / 2 + 0.5, 0))
    bpy.context.object.name = name
    bpy.context.object.data.materials.append(mat)

directory = ROOT / "target/surface-closures"
directory.mkdir(parents=True, exist_ok=True)
results = []
for lighting in ("world", "sun"):
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.7, 0.7, 0.7, 1) if lighting == "world" else (0, 0, 0, 1)
    if lighting == "sun":
        bpy.ops.object.light_add(type="SUN", rotation=(0.35, -0.5, 0.0))
        bpy.context.object.data.energy = 2
        bpy.context.object.data.angle = 0.15
    scene.render.engine = "CYCLES"
    scene.render.filepath = str(directory / f"{lighting}-cycles.exr")
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(scene.render.filepath, check_existing=False)
    reference = np.empty(resolution * resolution * 4, np.float32)
    img.pixels.foreach_get(reference)
    reference = reference.reshape(resolution, resolution, 4)
    scene.render.engine = "VIBRT"
    scene.vibrt_spp = SPP
    exported, blobs, textures = exporter.export_scene_to_memory(bpy.context.evaluated_depsgraph_get(), serialize=False)
    if lighting == "sun":
        sun = next(light for light in exported["lights"] if light["type"] == "sun")
        assert abs(sun["angle_rad"] - 0.075) < 1e-6
    (directory / f"{lighting}-export.json").write_text(json.dumps(exported, indent=2), encoding="utf-8")
    actual, _ = native.render(exported, [b.view(np.uint8) for b in blobs],
                             {"spp": SPP, "width": resolution, "height": resolution}, texture_arrays=textures)
    assert np.isfinite(actual).all()
    np.save(directory / f"{lighting}-cuda.npy", actual)
    for suffix, pixels in (("cycles", reference), ("cuda", actual)):
        out = bpy.data.images.new(suffix, width=resolution, height=resolution, float_buffer=True, alpha=True)
        out.pixels.foreach_set(pixels.ravel())
        scene.render.image_settings.file_format = "PNG"
        out.save_render(str(directory / f"{lighting}-{suffix}.png"), scene=scene)
        bpy.data.images.remove(out)
    scene.render.image_settings.file_format = "OPEN_EXR"
    for i, (name, _) in enumerate(cases):
        x, y = i % side * tile, i // side * tile
        ref = reference[y + 4:y + tile - 4, x + 4:x + tile - 4, :3].mean(axis=(0, 1))
        got = actual[y + 4:y + tile - 4, x + 4:x + tile - 4, :3].mean(axis=(0, 1))
        error = float(np.max(np.abs(ref - got) / np.maximum(ref, 0.04)))
        result = dict(name=name, lighting=lighting, cycles=ref.tolist(), cuda=got.tolist(), error=error)
        results.append(result)
        if error > 0.08:
            print("[surface difference]", result, flush=True)
    (directory / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
failures = [r for r in results if r["error"] > 0.08]
print(f"[surface-closures] {len(results)} comparisons, {len(failures)} differences", flush=True)
assert not failures
