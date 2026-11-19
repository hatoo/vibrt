"""Metallic floor lit by an emissive mesh, at several roughnesses, to isolate
which MIS side under-delivers:
  - rough ~0.02 (near-mirror): NEE can't find the narrow lobe -> delivery is
    BSDF-sampling-only (emission gate).
  - rough 0.5 (broad): NEE-dominant.
If the under is constant across roughness -> both sides / a shared issue.
If it appears only at high rough -> NEE side; only at low rough -> BSDF side.

Run: blender -b --factory-startup --python scripts/_diag_metal_rough_sweep.py -- <roughness>
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
rough = float(argv[argv.index("--") + 1]) if "--" in argv else 0.2

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

fm = bpy.data.materials.new("Floor"); fm.use_nodes = True
fb = fm.node_tree.nodes.get("Principled BSDF")
fb.inputs["Base Color"].default_value = (0.8, 0.6, 0.4, 1.0)
fb.inputs["Metallic"].default_value = 1.0
fb.inputs["Roughness"].default_value = rough
bpy.ops.mesh.primitive_plane_add(size=10)
bpy.context.active_object.data.materials.append(fm)

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
        s.vibrt_spp = 512
    else:
        s.cycles.samples = 512; s.cycles.use_adaptive_sampling = False
        s.cycles.sample_clamp_direct = 0; s.cycles.sample_clamp_indirect = 0
    p = OUT / f"mrs_{engine}.exr"; s.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(p)); img.colorspace_settings.name = "Linear Rec.709"
    ww, hh = int(img.size[0]), int(img.size[1])
    fl = np.empty(ww*hh*4, np.float32); img.pixels.foreach_get(fl); bpy.data.images.remove(img)
    return fl.reshape(hh, ww, 4)[..., :3].reshape(-1, 3)


v = render("vibrt"); c = render("cycles")
clum = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
floor = (clum > 0.001) & (clum < 15.0)
ratio = float((v[floor].mean(0) / np.maximum(c[floor].mean(0), 1e-9))[0])
print(f"RESULT metallic mesh rough={rough}: ratio_R={ratio:.4f} (n={int(floor.sum())})")
print("=== DONE ===", flush=True)
