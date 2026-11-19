"""Metallic floor lit by (a) an area light vs (b) an emissive mesh, to split
the metallic under-delivery into GGX BSDF vs mesh-light-MIS.

Run: blender -b --factory-startup --python scripts/_diag_metal_lights.py -- <area|mesh>
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
lt = argv[argv.index("--") + 1] if "--" in argv else "area"

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

# Metallic floor
fm = bpy.data.materials.new("Floor"); fm.use_nodes = True
fb = fm.node_tree.nodes.get("Principled BSDF")
fb.inputs["Base Color"].default_value = (0.8, 0.6, 0.4, 1.0)
fb.inputs["Metallic"].default_value = 1.0
fb.inputs["Roughness"].default_value = 0.2
bpy.ops.mesh.primitive_plane_add(size=10)
bpy.context.active_object.data.materials.append(fm)

if lt == "area":
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 3)); L = bpy.context.active_object
    L.data.energy = 1000.0; L.data.size = 1.5
else:
    em = bpy.data.materials.new("Emit"); em.use_nodes = True
    nt = em.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); en = nt.nodes.new("ShaderNodeEmission")
    en.inputs["Strength"].default_value = 20.0
    nt.links.new(en.outputs[0], out.inputs["Surface"])
    bpy.ops.mesh.primitive_plane_add(size=1.5, location=(0, 0, 3))
    q = bpy.context.active_object; q.data.materials.append(em); q.rotation_euler = (math.radians(180), 0, 0)

w = bpy.data.worlds.new("W"); w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
bpy.context.scene.world = w
bpy.ops.object.camera_add(location=(0, -6, 5), rotation=(math.radians(52), 0, 0))
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
        s.vibrt_spp = 256
    else:
        s.cycles.samples = 256; s.cycles.use_adaptive_sampling = False
        s.cycles.sample_clamp_direct = 0; s.cycles.sample_clamp_indirect = 0
    p = OUT / f"metL_{lt}_{engine}.exr"; s.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(p)); img.colorspace_settings.name = "Linear Rec.709"
    ww, hh = int(img.size[0]), int(img.size[1])
    fl = np.empty(ww*hh*4, np.float32); img.pixels.foreach_get(fl); bpy.data.images.remove(img)
    return fl.reshape(hh, ww, 4)[..., :3].reshape(-1, 3)


v = render("vibrt"); c = render("cycles")
clum = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
floor = (clum > 0.001) & (clum < 15.0)
r = v[floor].mean(0) / np.maximum(c[floor].mean(0), 1e-9)
print(f"RESULT metallic floor lit by {lt}: ratio v/c = {np.round(r,4)}  (n={int(floor.sum())})")
print("=== DONE ===", flush=True)
