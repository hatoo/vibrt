"""GPU area-light energy/MIS regression against independent area quadrature.

blender -b --factory-startup --python-exit-code 1 --python scripts/test_area_lights.py
CPU sampler check: compile and run scripts/test_rect_light_sampling.cpp.
"""
from pathlib import Path
import copy
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
s = bpy.context.scene
s.render.engine = "VIBRT"
s.render.resolution_x = s.render.resolution_y = 16
s.render.resolution_percentage = 100
s.vibrt_clamp_indirect = 0
s.cycles.max_bounces = 2
s.cycles.sample_clamp_direct = s.cycles.sample_clamp_indirect = 0
s.cycles.blur_glossy = 0
world = bpy.data.worlds.new("Black")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0
s.world = world
camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
s.collection.objects.link(camera)
camera.location = (0, 0, 3)
camera.data.angle = 0.002
s.camera = camera
bpy.ops.mesh.primitive_plane_add(size=100)
mat = bpy.data.materials.new("Lambert")
mat.use_nodes = True
mat.node_tree.nodes.clear()
diffuse = mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
diffuse.inputs["Color"].default_value = (0.7, 0.7, 0.7, 1)
output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
mat.node_tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
bpy.context.object.data.materials.append(mat)
description, blobs, textures = exporter.export_scene_to_memory(
    bpy.context.evaluated_depsgraph_get(), serialize=False)
blobs = [b.view(np.uint8) for b in blobs]


def rect(center, size, power=10, visible=False, back=False, two_sided=False):
    sign = 1 if back else -1
    x, y, z = center
    return {"type": "area_rect", "transform":
            [1, 0, 0, x, 0, sign, 0, y, 0, 0, sign, z, 0, 0, 0, 1],
            "size": list(size), "color": [1, 1, 1], "power": power,
            "camera_visible": int(visible), "two_sided": int(two_sided)}


def reference(lights):
    # Integrate Le * cos(receiver) * cos(emitter) / distance^2 over area.
    n = 700
    u, v = np.meshgrid((np.arange(n)+0.5)/n-0.5, (np.arange(n)+0.5)/n-0.5)
    total = 0.0
    for light in lights:
        transform = np.array(light["transform"]).reshape(4, 4)
        width, height = light["size"]
        point = (transform[:3, 3] + u[..., None]*width*transform[:3, 0]
                 + v[..., None]*height*transform[:3, 1])
        d2 = np.sum(point*point, axis=-1)
        wi = point / np.sqrt(d2)[..., None]
        cos_light = -np.sum(wi*transform[:3, 2], axis=-1)
        cos_light = np.abs(cos_light) if light["two_sided"] else np.maximum(cos_light, 0)
        total += float(np.mean(np.maximum(wi[..., 2], 0)*cos_light/d2)) * (
            0.7 * light["power"] / (math.pi*math.pi))
    return total


near = rect((0.3, 0.2, 0.5), (2, 3))
visible = rect((1.3, 0.1, 0.8), (1, 2), visible=True)
thin = rect((0.2, 0, 0.3), (6, 0.05))
far = rect((100, 100, 10000), (0.1, 0.4), power=1e9)
cases = {
    "near, hidden": [near],
    "visible (BSDF/NEE MIS)": [visible],
    "visible, last bounce (NEE only)": [visible],
    "thin": [thin],
    "distant (area fallback)": [far],
    "back face": [rect((0, 0, 1), (2, 2), back=True)],
    "two-sided back face": [rect((0, 0, 1), (2, 2), back=True, two_sided=True)],
    "zero power": [rect((0, 0, 1), (2, 2), power=0)],
    "mixed visible/hidden, near/far": [near, visible, far],
    "dark/back-facing alongside live": [visible, rect((0, 0, 1), (2, 2), power=0),
                                          rect((0, 0, 1), (2, 2), back=True)],
}
for name, lights in cases.items():
    scene = copy.deepcopy(description)
    scene["lights"] = lights
    options = {"spp": 4096, "denoise": False}
    if "last bounce" in name:
        options["max_depth"] = 1
    pixels, _ = native.render(scene, blobs, options,
                              texture_arrays=textures)
    assert np.isfinite(pixels).all(), name
    actual = float(pixels[..., :3].mean())
    expected = reference(lights)
    error = abs(actual-expected)/max(expected, 1e-8)
    print(f"[area-light] {name}: {actual:.6g} vs {expected:.6g}, error={error:.2%}", flush=True)
    assert error < 0.015, (name, actual, expected, error)
print(f"[area-light] {len(cases)} energy/MIS cases passed", flush=True)

# A dilute absorbing medium exercises phase/NEE MIS. Hiding the light only
# changes the estimator; it must not remove or duplicate scattered energy.
means = []
for visibility in (False, True):
    scene = copy.deepcopy(description)
    scene["lights"] = [rect((1.3, 0.1, 0.8), (1, 2), visible=visibility)]
    scene["world_volume"] = {"density": 0.2, "color": [1, 1, 1],
                             "absorption_color": [2, 2, 2], "anisotropy": 0.3}
    pixels, _ = native.render(scene, blobs, {"spp": 16384, "max_depth": 12},
                              texture_arrays=textures)
    assert np.isfinite(pixels).all()
    means.append(float(pixels[..., :3].mean()))
error = abs(means[0]-means[1])/means[0]
print(f"[area-light] volume hidden/visible: {means}, difference={error:.2%}", flush=True)
assert error < 0.02, means
