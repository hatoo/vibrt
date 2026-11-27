"""Render scene-dependent nodes against Cycles (headless Blender).

Uses analytic hit/miss fixtures as well as stochastic AO/Bevel. Artifacts:
target/scene-nodes/{cycles.exr,cuda.npy,results.json}.
Run with --factory-startup --python-exit-code 1 to avoid preloaded addons.
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
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 64
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.cycles.pixel_filter_type = "BOX"
scene.render.filter_size = 0.01
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.world = bpy.data.worlds.new("Black")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0

cases = []


def case(kind, output, **properties):
    mat = bpy.data.materials.new(f"{kind}.{output}.{properties}")
    mat.use_nodes = True
    t = mat.node_tree
    t.nodes.clear()
    out = t.nodes.new("ShaderNodeOutputMaterial")
    emit = t.nodes.new("ShaderNodeEmission")
    n = t.nodes.new(kind)
    for key, value in properties.items():
        setattr(n, key, value)
    t.links.new(n.outputs[output], emit.inputs["Color"])
    t.links.new(emit.outputs[0], out.inputs["Surface"])
    cases.append((mat.name, mat))
    return t, n


for pixel in (False, True):
    _, n = case("ShaderNodeWireframe", "Fac", use_pixel_size=pixel)
    n.inputs["Size"].default_value = 3 if pixel else 0.08
for local in (False, True):
    for output in ("AO", "Color"):
        _, n = case("ShaderNodeAmbientOcclusion", output, only_local=local, samples=16)
        n.inputs["Distance"].default_value = 0.7
        n.inputs["Color"].default_value = (0.3, 0.6, 0.9, 1)
for local in (False, True):
    for output in ("Is Hit", "Self Hit", "Hit Distance", "Hit Normal"):
        t, n = case("ShaderNodeRaycast", output, only_local=local)
        d = t.nodes.new("ShaderNodeCombineXYZ")
        d.inputs["X"].default_value = 1
        d.inputs["Z"].default_value = 0.15
        t.links.new(d.outputs[0], n.inputs["Direction"])
for output in ("Self Hit", "Hit Distance"):
    t, n = case("ShaderNodeRaycast", output, only_local=True)
    d = t.nodes.new("ShaderNodeCombineXYZ")
    d.inputs["Z"].default_value = -1
    t.links.new(d.outputs[0], n.inputs["Direction"])
for radius in (0, 0.15):
    _, n = case("ShaderNodeBevel", "Normal", samples=16)
    n.inputs["Radius"].default_value = radius

if os.environ.get("VIBRT_SCENE_NODE_FILTER"):
    cases = [
        (name, mat)
        for name, mat in cases
        if os.environ["VIBRT_SCENE_NODE_FILTER"] in name
    ]
side = math.ceil(math.sqrt(len(cases)))
tile = 40
resolution = side * tile
scene.render.resolution_x = scene.render.resolution_y = resolution
camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
scene.collection.objects.link(camera)
scene.camera = camera
camera.data.angle = math.pi / 3
camera.location = (0, 0, side / 2 / math.tan(camera.data.angle / 2))
black = bpy.data.materials.new("Occluder")
black.use_nodes = True
black.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
    0,
    0,
    0,
    1,
)
shared_cube = None
for i, (name, mat) in enumerate(cases):
    x, y = i % side - side / 2 + 0.5, i // side - side / 2 + 0.5
    # The closed box gives Bevel actual adjacent faces on the same object.
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, -0.25))
    obj = bpy.context.object
    obj.scale.z = 0.5
    # Linked instances share primitive IDs; local and self-hit queries must
    # identify the instance, not merely the mesh/primitive. Keep nonuniform scale.
    if shared_cube is None:
        shared_cube = obj.data
        shared_cube.materials.append(mat)
    else:
        unused = obj.data
        obj.data = shared_cube
        bpy.data.meshes.remove(unused)
    obj.material_slots[0].link = "OBJECT"
    obj.material_slots[0].material = mat
    if "Bevel" not in name and "Wireframe" not in name:
        mesh = bpy.data.meshes.new("Wall")
        mesh.from_pydata(
            [
                (x + 0.38, y - 0.48, -0.1),
                (x + 0.38, y + 0.48, -0.1),
                (x + 0.38, y + 0.48, 0.7),
                (x + 0.38, y - 0.48, 0.7),
            ],
            [],
            [(0, 1, 2, 3)],
        )
        mesh.materials.append(black)
        wall = bpy.data.objects.new("Wall", mesh)
        scene.collection.objects.link(wall)

directory = ROOT / "target/scene-nodes"
directory.mkdir(parents=True, exist_ok=True)
scene.render.filepath = str(directory / "cycles.exr")
bpy.ops.render.render(write_still=True)
ref_image = bpy.data.images.load(scene.render.filepath, check_existing=False)
reference = np.empty(resolution * resolution * 4, np.float32)
ref_image.pixels.foreach_get(reference)
reference = reference.reshape(resolution, resolution, 4)
scene.render.engine = "VIBRT"
desc, blobs, textures = exporter.export_scene_to_memory(
    bpy.context.evaluated_depsgraph_get(), serialize=False
)
actual, _ = native.render(
    desc,
    [b.view(np.uint8) for b in blobs],
    {"spp": 64, "width": resolution, "height": resolution},
    texture_arrays=textures,
)
assert np.isfinite(actual).all()
np.save(directory / "cuda.npy", actual)
rows = []
for i, (name, _) in enumerate(cases):
    x, y = i % side * tile, i // side * tile
    # Avoid silhouettes and wall projection; compare a region to reduce MC noise.
    crop = (slice(y + 8, y + 24), slice(x + 8, x + 20), slice(0, 3))
    if "Bevel" in name:
        crop = (slice(y + 12, y + 28), slice(x + 2, x + 7), slice(0, 3))
    expected, rendered = reference[crop], actual[crop]
    error = float(np.max(np.abs(expected.mean((0, 1)) - rendered.mean((0, 1)))))
    tolerance = 0.06 if "Ambient" in name or "Bevel" in name else 0.025
    rows.append(
        dict(
            name=name,
            error=error,
            tolerance=tolerance,
            cycles=expected.mean((0, 1)).tolist(),
            cuda=rendered.mean((0, 1)).tolist(),
        )
    )
(directory / "results.json").write_text(json.dumps(rows, indent=2))
failed = [r for r in rows if r["error"] > r["tolerance"]]
for r in failed:
    print(r, flush=True)
print(f"[scene-nodes] {len(rows)} cases, {len(failed)} failures", flush=True)
assert not failed
