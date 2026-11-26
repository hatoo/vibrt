"""Orientation check: compare the elevation profile (per-row mean) of vibrt's
baked sky against a canonical Cycles EQUIRECTANGULAR render of the same
sky-alone world. A canonical equirect has zenith at the TOP row (v=0) and
nadir at the BOTTOM. If vibrt's baked profile is flipped/shifted, the bake
camera orientation is the bug.
"""
import sys, math
from pathlib import Path
import numpy as np
import bpy

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "blender"))
for _m in list(sys.modules):
    if _m == "vibrt_blender" or _m.startswith("vibrt_blender."):
        del sys.modules[_m]
import vibrt_blender  # noqa
from vibrt_blender.export_context import ExportContext
export_context = ExportContext()
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass
from vibrt_blender import exporter as EXP

scene = bpy.context.scene
world = scene.world
sky_node = next(n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeTexSky")

# --- vibrt bake (sky node alone) ---
EXP.prebake_sky_envmaps_for_world(world, w=2048, h=1024, context=export_context)
key = EXP._sky_node_alone_cache_key(world, sky_node)
sky_rgb, bw, bh, sun = export_context.sky_maps[key]
vib = np.asarray(sky_rgb, dtype=np.float32).reshape(bh, bw, 3)
# vibrt bake storage note: foreach_get bottom-up; the diag treats array row 0
# per how the kernel reads it (v=0 -> row 0). Report as-stored.

# --- canonical Cycles equirect of the SAME sky-alone world ---
tmp = bpy.data.scenes.new("__canon_sky")
tw = bpy.data.worlds.new("__canon_world")
tw.use_nodes = True
nt = tw.node_tree
nt.nodes.clear()
outn = nt.nodes.new("ShaderNodeOutputWorld")
bg = nt.nodes.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = 1.0
sky2 = nt.nodes.new("ShaderNodeTexSky")
for a in ("sky_type", "sun_disc", "sun_size", "sun_intensity", "sun_elevation",
          "sun_rotation", "altitude", "air_density", "dust_density",
          "ozone_density", "ground_albedo", "turbidity"):
    if hasattr(sky_node, a) and hasattr(sky2, a):
        try:
            setattr(sky2, a, getattr(sky_node, a))
        except Exception:
            pass
nt.links.new(sky2.outputs["Color"], bg.inputs["Color"])
nt.links.new(bg.outputs["Background"], outn.inputs["Surface"])
tmp.world = tw
tmp.render.engine = "CYCLES"
tmp.cycles.samples = 64
tmp.cycles.use_adaptive_sampling = False
tmp.render.resolution_x = 512
tmp.render.resolution_y = 256
tmp.render.image_settings.file_format = "OPEN_EXR"
tmp.render.image_settings.color_depth = "32"
tmp.view_settings.view_transform = "Raw"
tmp.view_settings.look = "None"
camd = bpy.data.cameras.new("__c"); camd.type = "PANO"
camd.panorama_type = "EQUIRECTANGULAR"
camo = bpy.data.objects.new("__c", camd)
camo.location = (0, 0, 0)
# Blender equirect default: camera -Z forward, +Y up. To get zenith(+Z world)
# at top row, point the camera's up toward world +Z: rotate +90deg about X.
camo.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
tmp.collection.objects.link(camo)
tmp.camera = camo
import tempfile, os
fd, p = tempfile.mkstemp(suffix=".exr"); os.close(fd)
tmp.render.filepath = p
bpy.ops.render.render(scene=tmp.name, write_still=True)
im = bpy.data.images.load(p)
im.colorspace_settings.name = "Non-Color"
cw, chh = int(im.size[0]), int(im.size[1])
flat = np.empty(cw * chh * 4, dtype=np.float32)
im.pixels.foreach_get(flat)
canon = flat.reshape(chh, cw, 4)[..., :3][::-1].copy()  # top-down: row0=top
bpy.data.images.remove(im)
os.unlink(p)

print(f"vibrt bake {bw}x{bh}, canon equirect {cw}x{chh}")
print("\nelevation profile (per-row mean, top->bottom, 9 samples):")
print(" frac   vibrt(row mean)            canon(row mean)")
for f in (0.02, 0.12, 0.25, 0.4, 0.5, 0.6, 0.75, 0.88, 0.98):
    vr = vib[int(f * (bh - 1))].mean(axis=0)
    cr = canon[int(f * (chh - 1))].mean(axis=0)
    print(f" {f:.2f}  {np.round(vr,4)}   {np.round(cr,4)}")

# Also report top vs bottom to spot a flip.
print(f"\nvibrt  row0(top)={np.round(vib[0].mean(0),4)}  rowN(bot)={np.round(vib[-1].mean(0),4)}")
print(f"canon  row0(top)={np.round(canon[0].mean(0),4)}  rowN(bot)={np.round(canon[-1].mean(0),4)}")
print("=== DONE ===", flush=True)
