"""Check Mix/Add BSDF energy against independently rendered component BSDFs.

Run with headless Blender, like scripts/test_shader_graphs.py. Coplanar tiles
under a constant world make material mixing linear. An orthographic camera
gives every tile the same incoming direction, so independent components and
their mixtures can share one render and one pipeline compilation.
"""
from copy import deepcopy
import json
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
scene.render.engine = "VIBRT"
scene.cycles.max_bounces = 2
world = bpy.data.worlds.new("Constant radiance")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1
scene.world = world
camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
scene.collection.objects.link(camera)
scene.camera = camera
camera.location = (0, 0, 3)
camera.data.type = "ORTHO"


def material(color, normal=(0, 0, 1), **parameters):
    return dict(base_color=color, shader_graph={
        "nodes": [{"op": "constant", "params": [*normal, 1]}],
        "outputs": {"normal": 0},
    }, **parameters)


def mix(left, right, factor=0.5, add=False):
    return dict(base_color=[0, 0, 0], left_subtree=deepcopy(left),
                secondary=deepcopy(right), mix_fac=factor, add_shader=add)


components = {
    "diffuse": material([0.2, 0.4, 0.7], pure_diffuse=True, roughness=0),
    # The tilted normal gives this leaf a different view cosine from its sibling.
    "coated": material([0.7, 0.3, 0.1], normal=(0.3, 0, 0.9539392),
                       roughness=0.28, metallic=0.65, coat_weight=0.5,
                       coat_roughness=0.2, coat_ior=1.45),
    "glass": material([0.95, 0.92, 0.9], transmission=1, roughness=0.22, ior=1.45),
    "alpha": material([0.3, 0.55, 0.1], pure_diffuse=True, alpha_blend=0.4),
    "hair": material([0.2, 0.3, 0.4], hair_weight=1,
                     hair_roughness_u=0.3, hair_roughness_v=0.5),
    "translucent": material([0.3, 0.6, 0.2], pure_diffuse=True, translucent_weight=1),
}
directory = ROOT / "target/bsdf-mixtures"
directory.mkdir(parents=True, exist_ok=True)


a, b, c, d = (components[name] for name in ("diffuse", "coated", "glass", "alpha"))
cases = [
    ("mixed", mix(a, b, 0.3), {"diffuse": 0.7, "coated": 0.3}),
    ("added", mix(a, b, add=True), {"diffuse": 1, "coated": 1}),
    ("nested", mix(mix(a, c, 0.65), b, 0.2),
     {"diffuse": 0.8 * 0.35, "glass": 0.8 * 0.65, "coated": 0.2}),
    ("transparent", mix(b, d, 0.55), {"coated": 0.45, "alpha": 0.55}),
    ("added-glass", mix(b, c, add=True), {"coated": 1, "glass": 1}),
    ("mixed-hair", mix(a, components["hair"], 0.6), {"diffuse": 0.4, "hair": 0.6}),
    ("mixed-translucent", mix(b, components["translucent"], 0.4),
     {"coated": 0.6, "translucent": 0.4}),
    ("zero-weight", mix(a, b, 0), {"diffuse": 1}),
    ("full-weight", mix(a, b, 1), {"coated": 1}),
]
materials = dict(components)
materials.update((name, mat) for name, mat, _ in cases)
side, tile = 4, 32
assert len(materials) <= side * side
scene.render.resolution_x = scene.render.resolution_y = side * tile
camera.data.ortho_scale = side
for index, name in enumerate(materials):
    bpy.ops.mesh.primitive_plane_add(size=1, location=(
        index % side - side / 2 + 0.5, index // side - side / 2 + 0.5, 0))
    placeholder = bpy.data.materials.new(name)
    # Preserve the fixture ID through export without relying on material order.
    placeholder.use_nodes = True
    placeholder.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
        (index + 1) / 16, 0, 0, 1)
    bpy.context.object.data.materials.append(placeholder)
description, blobs, textures = exporter.export_scene_to_memory(
    bpy.context.evaluated_depsgraph_get(), serialize=False
)
names = list(materials)
description["materials"] = [materials[names[round(mat["base_color"][0] * 16) - 1]]
                            for mat in description["materials"]]
pixels, _ = native.render(description, [blob.view(np.uint8) for blob in blobs],
                          {"spp": 8192, "denoise": False}, texture_arrays=textures)
assert np.isfinite(pixels).all()
np.save(directory / "atlas.npy", pixels)
measurements = {}
for index, name in enumerate(materials):
    x, y = index % side * tile, index // side * tile
    measurements[name] = pixels[y + 4:y + tile - 4, x + 4:x + tile - 4, :3].mean(
        axis=(0, 1), dtype=np.float64)
reference = {name: measurements[name] for name in components}
# Independent tile samples include Russian roulette; allow Monte Carlo error
# in the white-furnace check as well as in the mixture comparisons below.
np.testing.assert_allclose(reference["diffuse"], [0.2, 0.4, 0.7], atol=1e-3, rtol=0)
results = []
for name, mat, weights in cases:
    expected = sum(reference[key] * weight for key, weight in weights.items())
    actual = measurements[name]
    error = float(np.max(np.abs(actual - expected) / np.maximum(0.1, np.abs(expected))))
    results.append(dict(name=name, expected=expected.tolist(), actual=actual.tolist(),
                        relative_error=error))
    print(f"[bsdf-mixtures] {name}: relative error {error:.4%}", flush=True)

(directory / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
# Independent Monte Carlo sequences: compare average radiance, not individual pixels.
assert max(result["relative_error"] for result in results) < 0.025, results
print(f"[bsdf-mixtures] PASS {len(cases)} cases", flush=True)
