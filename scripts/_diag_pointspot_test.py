"""Test plain point/spot/area delta-light delivery (no IES) vs Cycles in a
controlled scene, to isolate whether ies_light's under-delivery is the core
delta-light delivery or the IES coefficient.

Run: blender -b --factory-startup --python scripts/_diag_pointspot_test.py -- <point|spot|area>
"""
import sys, math
from pathlib import Path
import numpy as np
import bpy

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

argv = sys.argv
kind = argv[argv.index("--") + 1] if "--" in argv else "point"

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

m = bpy.data.materials.new("Floor"); m.use_nodes = True
b = m.node_tree.nodes.get("Principled BSDF")
b.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
b.inputs["Roughness"].default_value = 1.0; b.inputs["Metallic"].default_value = 0.0
try: b.inputs["Specular IOR Level"].default_value = 0.0
except Exception: pass
bpy.ops.mesh.primitive_plane_add(size=10)
bpy.context.active_object.data.materials.append(m)

if kind == "point":
    bpy.ops.object.light_add(type='POINT', location=(0, 0, 3)); L = bpy.context.active_object
    L.data.energy = 1000.0; L.data.shadow_soft_size = 0.05
elif kind == "spot":
    bpy.ops.object.light_add(type='SPOT', location=(0, 0, 3)); L = bpy.context.active_object
    L.data.energy = 1000.0; L.data.spot_size = math.radians(60); L.data.spot_blend = 0.15
    L.data.shadow_soft_size = 0.05
else:
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 3)); L = bpy.context.active_object
    L.data.energy = 1000.0; L.data.size = 0.5

w = bpy.data.worlds.new("W"); w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
bpy.context.scene.world = w
bpy.ops.object.camera_add(location=(0, -6, 6), rotation=(math.radians(48), 0, 0))
bpy.context.scene.camera = bpy.context.active_object

s = bpy.context.scene
s.render.resolution_x = 300; s.render.resolution_y = 300
s.render.image_settings.file_format = "OPEN_EXR"; s.render.image_settings.color_depth = "32"
s.view_settings.view_transform = "Raw"; s.view_settings.look = "None"
s.view_settings.exposure = 0.0; s.view_settings.gamma = 1.0
s.render.use_compositing = False
OUT = HERE / "test_scenes" / "_eng_diff"; OUT.mkdir(parents=True, exist_ok=True)


def render(engine):
    s.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        s.vibrt_spp = 128
    else:
        s.cycles.samples = 128; s.cycles.use_adaptive_sampling = False
        s.cycles.sample_clamp_direct = 0; s.cycles.sample_clamp_indirect = 0
    p = OUT / f"pslight_{kind}_{engine}.exr"; s.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(p)); img.colorspace_settings.name = "Linear Rec.709"
    ww, hh = int(img.size[0]), int(img.size[1])
    fl = np.empty(ww*hh*4, np.float32); img.pixels.foreach_get(fl); bpy.data.images.remove(img)
    return fl.reshape(hh, ww, 4)[..., :3].reshape(-1, 3)


v = render("vibrt"); c = render("cycles")
# only lit pixels (exclude black background)
clum = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
mask = clum > 0.001
print(f"\n=== {kind} light (no IES) ===")
print(f"lit-pixel ratio v/c = {np.round(v[mask].mean(0)/np.maximum(c[mask].mean(0),1e-9),4)}  (n={int(mask.sum())})")
print(f"  vibrt lit mean = {np.round(v[mask].mean(0),4)}  cyc lit mean = {np.round(c[mask].mean(0),4)}")
print("=== DONE ===", flush=True)
