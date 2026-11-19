"""Test emissive-mesh lighting (mesh-light NEE) of diffuse vs glossy floors,
and direct emissive-mesh view, vs Cycles — isolates bmw27's emissive-mesh
under-delivery (bmw27 is lit entirely by emissive meshes).

Run: blender -b --factory-startup --python scripts/_diag_meshlight_test.py -- <diffuse|glossy|glass>
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
kind = argv[argv.index("--") + 1] if "--" in argv else "diffuse"

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

# Floor material
fm = bpy.data.materials.new("Floor"); fm.use_nodes = True
fb = fm.node_tree.nodes.get("Principled BSDF")
fb.inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1.0)
fb.inputs["Metallic"].default_value = 0.0
if kind == "glossy":
    fb.inputs["Roughness"].default_value = 0.15
elif kind == "metallic":
    fb.inputs["Roughness"].default_value = 0.2
    fb.inputs["Metallic"].default_value = 1.0
    fb.inputs["Base Color"].default_value = (0.8, 0.6, 0.4, 1.0)
elif kind == "coat":
    fb.inputs["Roughness"].default_value = 0.3
    try: fb.inputs["Coat Weight"].default_value = 1.0
    except Exception: pass
else:
    fb.inputs["Roughness"].default_value = 1.0
    try: fb.inputs["Specular IOR Level"].default_value = 0.0
    except Exception: pass
bpy.ops.mesh.primitive_plane_add(size=10)
bpy.context.active_object.data.materials.append(fm)

# Optional glass panel between camera and emitter (bmw27 headlight cover).
if kind == "glass":
    gm = bpy.data.materials.new("Glass"); gm.use_nodes = True
    gb = gm.node_tree.nodes.get("Principled BSDF")
    gb.inputs["Transmission Weight"].default_value = 1.0
    gb.inputs["Roughness"].default_value = 0.0
    gb.inputs["IOR"].default_value = 1.45
    bpy.ops.mesh.primitive_plane_add(size=3, location=(0, 0, 1.5))
    gp = bpy.context.active_object; gp.data.materials.append(gm)
    gp.rotation_euler = (math.radians(90), 0, 0)  # vertical panel facing camera

# Emissive mesh quad above (the "light")
em = bpy.data.materials.new("Emit"); em.use_nodes = True
nt = em.node_tree; nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
emn = nt.nodes.new("ShaderNodeEmission")
emn.inputs["Strength"].default_value = 20.0
emn.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
nt.links.new(emn.outputs[0], out.inputs["Surface"])
bpy.ops.mesh.primitive_plane_add(size=1.5, location=(0, 0, 3))
q = bpy.context.active_object; q.data.materials.append(em)
q.rotation_euler = (math.radians(180), 0, 0)  # face down

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
    p = OUT / f"ml_{kind}_{engine}.exr"; s.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(p)); img.colorspace_settings.name = "Linear Rec.709"
    ww, hh = int(img.size[0]), int(img.size[1])
    fl = np.empty(ww*hh*4, np.float32); img.pixels.foreach_get(fl); bpy.data.images.remove(img)
    return fl.reshape(hh, ww, 4)[..., :3].reshape(-1, 3)


v = render("vibrt"); c = render("cycles")
clum = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
# Exclude the emitter quad itself (very bright, >15) to isolate floor lighting.
floor_m = (clum > 0.001) & (clum < 15.0)
emit_m = clum >= 15.0
print(f"\n=== mesh-light {kind} floor ===")
print(f"floor lit ratio v/c = {np.round(v[floor_m].mean(0)/np.maximum(c[floor_m].mean(0),1e-9),4)}  (n={int(floor_m.sum())})")
if emit_m.sum() > 0:
    print(f"emitter direct ratio = {np.round(v[emit_m].mean(0)/np.maximum(c[emit_m].mean(0),1e-9),4)}  (n={int(emit_m.sum())})")
print("=== DONE ===", flush=True)
