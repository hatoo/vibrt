"""Cycles comparison for Toon, Holdout, Ray Portal and custom AOV output."""

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
scene.cycles.sample_clamp_direct = 0
scene.cycles.sample_clamp_indirect = 0
scene.render.film_transparent = True
scene.cycles.pixel_filter_type = "BOX"
scene.render.filter_size = 0.01
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
world = bpy.data.worlds.new("White")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
scene.world = world
for name, kind in (("CustomColor", "COLOR"), ("CustomValue", "VALUE")):
    a = bpy.context.view_layer.aovs.add()
    a.name = name
    a.type = kind
cases = []
for kind, options in [
    ("ShaderNodeBsdfToon", ("DIFFUSE", 0)),
    ("ShaderNodeBsdfToon", ("DIFFUSE", 0.3)),
    ("ShaderNodeBsdfToon", ("GLOSSY", 0.2)),
    ("ShaderNodeHoldout", None),
    ("ShaderNodeBsdfRayPortal", None),
    ("ShaderNodeHoldout", 0.5),
    ("ShaderNodeBsdfHairPrincipled", "COLOR"),
    ("ShaderNodeBsdfHairPrincipled", "MELANIN"),
    ("ShaderNodeBsdfHairPrincipled", "ABSORPTION"),
    ("ShaderNodeSubsurfaceScattering", "BURLEY"),
]:
    mat = bpy.data.materials.new(f"{kind}-{options}")
    mat.use_nodes = True
    t = mat.node_tree
    t.nodes.clear()
    n = t.nodes.new(kind)
    out = t.nodes.new("ShaderNodeOutputMaterial")
    if kind == "ShaderNodeSubsurfaceScattering":
        n.falloff = options
        n.inputs["Color"].default_value = (0.2, 0.4, 0.7, 1)
        n.inputs["Scale"].default_value = 0.2
    if kind == "ShaderNodeBsdfHairPrincipled":
        n.model = "CHIANG"
        n.parametrization = options
        sockets = {s.name: s for s in n.inputs}
        sockets["Color"].default_value = (0.2, 0.4, 0.7, 1)
        sockets["Melanin"].default_value = 0.3
        sockets["Absorption Coefficient"].default_value = (0.7, 0.4, 0.2)
    if kind == "ShaderNodeBsdfToon":
        n.component = options[0]
        n.inputs["Smooth"].default_value = options[1]
        n.inputs["Color"].default_value = (0.2, 0.5, 0.8, 1)
    if kind == "ShaderNodeBsdfRayPortal":
        n.inputs["Color"].default_value = (0.2, 0.5, 0.8, 1)
        for field, vector in (("Position", (0, 0, 2)), ("Direction", (0, 0, 1))):
            v = t.nodes.new("ShaderNodeCombineXYZ")
            for i, x in enumerate(vector):
                v.inputs[i].default_value = x
            t.links.new(v.outputs[0], n.inputs[field])
    leaf = n
    if kind == "ShaderNodeHoldout" and options == 0.5:
        other = t.nodes.new("ShaderNodeBsdfDiffuse")
        other.inputs["Color"].default_value = (0.2, 0.5, 0.8, 1)
        mix = t.nodes.new("ShaderNodeMixShader")
        t.links.new(n.outputs[0], mix.inputs[1])
        t.links.new(other.outputs[0], mix.inputs[2])
        leaf = mix
    t.links.new(leaf.outputs[0], out.inputs["Surface"])
    for name, value, field in (
        ("CustomColor", (0.13, 0.27, 0.53, 1), "Color"),
        ("CustomValue", 0.37, "Value"),
    ):
        a = t.nodes.new("ShaderNodeOutputAOV")
        a.aov_name = name
        a.inputs[field].default_value = value
    cases.append((mat.name, mat))
side = math.ceil(math.sqrt(len(cases)))
tile = 32
resolution = side * tile
scene.render.resolution_x = scene.render.resolution_y = resolution
camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
scene.collection.objects.link(camera)
scene.camera = camera
camera.data.angle = math.pi / 3
camera.location = (0, 0, side / 2 / math.tan(camera.data.angle / 2))
for i, (_, mat) in enumerate(cases):
    bpy.ops.mesh.primitive_plane_add(
        size=1, location=(i % side - side / 2 + 0.5, i // side - side / 2 + 0.5, 0)
    )
    bpy.context.object.data.materials.append(mat)
directory = ROOT / "target/extended-closures"
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
aovs = {}
actual, _ = native.render(
    desc,
    [b.view(np.uint8) for b in blobs],
    {"spp": 2048, "width": resolution, "height": resolution},
    texture_arrays=textures,
    aov_outputs=aovs,
)
np.save(directory / "cuda.npy", actual)
assert np.isfinite(actual).all()
rows = []
for i, (name, _) in enumerate(cases):
    x, y = i % side * tile, i // side * tile
    crop = (slice(y + 8, y + 24), slice(x + 8, x + 24))
    expected = reference[crop].mean((0, 1))
    rendered = actual[crop].mean((0, 1))
    error = float(np.max(abs(expected - rendered)))
    rows.append(
        dict(name=name, cycles=expected.tolist(), vibrt=rendered.tolist(), error=error)
    )
    # AOVs are camera visibility data, independent of emitted/received light.
    assert (
        np.max(
            abs(aovs["CustomColor"][crop][:, :, :3].mean((0, 1)) - [0.13, 0.27, 0.53])
        )
        < 0.005
    )
    assert abs(float(aovs["CustomValue"][crop][:, :, 0].mean()) - 0.37) < 0.005
(directory / "results.json").write_text(json.dumps(rows, indent=2))
failed = [r for r in rows if r["error"] > 0.045]
for r in failed:
    print(r, flush=True)
print(
    f"[extended-closures] {len(rows)} closures and 2 AOVs; {len(failed)} differences",
    flush=True,
)
assert not failed
