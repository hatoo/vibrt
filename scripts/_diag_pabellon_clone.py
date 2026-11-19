"""Test whether _prebake_sky_node_alone's CLONED sky reproduces pabellon's
ACTUAL sky. Renders the REAL world (fac forced to 0 => sky Background.003
only) through an equirect camera, and compares its elevation profile to
vibrt's baked (cloned) sky. Divergence => clone bug.
"""
import sys, math, tempfile, os
from pathlib import Path
import numpy as np
import bpy

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "blender"))
for _m in list(sys.modules):
    if _m == "vibrt_blender" or _m.startswith("vibrt_blender."):
        del sys.modules[_m]
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass
from vibrt_blender import exporter as EXP

scene = bpy.context.scene
world = scene.world
sky_node = next(n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeTexSky")

# vibrt bake (cloned sky) at strength captured in layer, but bake itself is
# strength=1.0 sky. Get raw bake.
EXP.prebake_sky_envmaps_for_world(world, w=2048, h=1024)
key = EXP._sky_node_alone_cache_key(world, sky_node)
sky_rgb, bw, bh, _sun = EXP._SKY_BAKE_CACHE[key]
vib = np.asarray(sky_rgb, dtype=np.float32).reshape(bh, bw, 3)

# Render pabellon's REAL world sky (Background.003) via equirect camera.
# Temporarily rewire the world output to Background.003 directly (strength 1.0
# so it's comparable to the strength-1.0 bake).
nt = world.node_tree
outw = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputWorld")
bg003 = None
for n in nt.nodes:
    if n.bl_idname == "ShaderNodeBackground":
        # find the one fed by the Sky Texture
        cs = n.inputs.get("Color")
        if cs and cs.is_linked and cs.links[0].from_node == sky_node:
            bg003 = n
assert bg003 is not None, "could not find sky Background"
orig_strength = bg003.inputs["Strength"].default_value
orig_link_from = outw.inputs["Surface"].links[0].from_socket if outw.inputs["Surface"].is_linked else None
bg003.inputs["Strength"].default_value = 1.0
nt.links.new(bg003.outputs["Background"], outw.inputs["Surface"])

tmp = bpy.data.scenes.new("__real_sky")
tmp.world = world
tmp.render.engine = "CYCLES"
tmp.cycles.samples = 64
tmp.cycles.use_adaptive_sampling = False
tmp.render.resolution_x = 512
tmp.render.resolution_y = 256
tmp.render.image_settings.file_format = "OPEN_EXR"
tmp.render.image_settings.color_depth = "32"
tmp.view_settings.view_transform = "Raw"
tmp.view_settings.look = "None"
camd = bpy.data.cameras.new("__rc"); camd.type = "PANO"
camd.panorama_type = "EQUIRECTANGULAR"
camo = bpy.data.objects.new("__rc", camd)
camo.location = (0, 0, 0)
camo.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
tmp.collection.objects.link(camo)
tmp.camera = camo
fd, p = tempfile.mkstemp(suffix=".exr"); os.close(fd)
tmp.render.filepath = p
bpy.ops.render.render(scene=tmp.name, write_still=True)
im = bpy.data.images.load(p)
im.colorspace_settings.name = "Non-Color"
cw, chh = int(im.size[0]), int(im.size[1])
flat = np.empty(cw * chh * 4, dtype=np.float32)
im.pixels.foreach_get(flat)
real = flat.reshape(chh, cw, 4)[..., :3][::-1].copy()
bpy.data.images.remove(im); os.unlink(p)

print(f"vibrt bake {bw}x{bh}, REAL sky equirect {cw}x{chh}")
print(" frac   vibrt(cloned bake)         REAL sky (native)          ratio")
for f in (0.02, 0.12, 0.25, 0.4, 0.5, 0.6, 0.75, 0.88, 0.98):
    vr = vib[int(f * (bh - 1))].mean(axis=0)
    rr = real[int(f * (chh - 1))].mean(axis=0)
    print(f" {f:.2f}  {np.round(vr,4)}   {np.round(rr,4)}   {np.round(vr/np.maximum(rr,1e-6),3)}")
vfull = vib.reshape(-1, 3).mean(0)
rfull = real.reshape(-1, 3).mean(0)
print(f"\nfull-sphere  vibrt bake={np.round(vfull,4)}  REAL sky={np.round(rfull,4)}  ratio={np.round(vfull/np.maximum(rfull,1e-6),3)}")
print("=== DONE ===", flush=True)
