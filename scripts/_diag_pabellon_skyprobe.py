"""Decisive sky-orientation probe for pabellon.

1. Force world MixShader fac=0 (sky only).
2. Render Cycles NATIVE (perspective) sky at low res -> native[y,x].
3. Bake vibrt's sky (prebake) -> equirect envmap (layer a pixels).
4. For each rendered pixel, compute its world ray direction and sample the
   baked equirect with vibrt's standard convention -> sampled[y,x].
5. Compare native vs sampled. Match => kernel read diverges; mismatch =>
   the bake doesn't represent the native sky (bake-setup bug).
"""
import sys, math
from pathlib import Path
import numpy as np
import bpy
from mathutils import Vector

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
mix = next(n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeMixShader")
mix.inputs[0].default_value = 0.0  # sky only (first shader = sky, strength 0.5)

# --- 3. Bake vibrt's sky (populates export_context.sky_maps) ---
EXP.prebake_sky_envmaps_for_world(world, w=2048, h=1024, context=export_context)
# Find the sky-alone bake in the cache.
sky_node = next(n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeTexSky")
key = EXP._sky_node_alone_cache_key(world, sky_node)
entry = export_context.sky_maps.get(key)
if entry is None:
    print("NO BAKE ENTRY for", key)
    for k in export_context.sky_maps:
        print("  have:", k)
    sys.exit(1)
sky_rgb, bw, bh, sun = entry
sky_rgb = np.asarray(sky_rgb, dtype=np.float32).reshape(bh, bw, 3)
print(f"baked sky envmap {bw}x{bh} mean={np.round(sky_rgb.reshape(-1,3).mean(0),4)} sun_extracted={sun is not None}")

STRENGTH_A = 0.5  # Background.003 strength; native fac=0 outputs 0.5*sky


def sample_equirect(rgb, d):
    # vibrt convention (matches sample_equirect_layer / sample_layer_at_dir).
    h_, w_, _ = rgb.shape
    theta = math.acos(min(max(d[2], -1.0), 1.0))
    phi = math.atan2(d[1], d[0])
    u = phi * (0.5 / math.pi)
    if u < 0:
        u += 1.0
    v = theta / math.pi
    fx = u * w_ - 0.5
    fy = v * h_ - 0.5
    x0 = int(math.floor(fx)); y0 = int(math.floor(fy))
    dx = fx - x0; dy = fy - y0
    def wr(i, m): r = i % m; return r + m if r < 0 else r
    x1 = wr(x0 + 1, w_); x0 = wr(x0, w_)
    y0c = max(0, min(h_ - 1, y0)); y1c = max(0, min(h_ - 1, y0 + 1))
    c00 = rgb[y0c, x0]; c10 = rgb[y0c, x1]; c01 = rgb[y1c, x0]; c11 = rgb[y1c, x1]
    c0 = c00 * (1 - dx) + c10 * dx
    c1 = c01 * (1 - dx) + c11 * dx
    return c0 * (1 - dy) + c1 * dy


# --- 2. Render Cycles native sky-only ---
PCT = 20
scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.cycles.use_adaptive_sampling = False
scene.render.resolution_percentage = PCT
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "32"
scene.view_settings.view_transform = "Raw"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.render.use_compositing = False
outp = HERE / "test_scenes" / "_eng_diff" / "pab_skyprobe_cycles.exr"
scene.render.filepath = str(outp)
bpy.ops.render.render(write_still=True)

img = bpy.data.images.load(str(outp))
img.colorspace_settings.name = "Linear Rec.709"
rw, rh = int(img.size[0]), int(img.size[1])
flat = np.empty(rw * rh * 4, dtype=np.float32)
img.pixels.foreach_get(flat)
native = flat.reshape(rh, rw, 4)[..., :3][::-1].copy()  # top-down
bpy.data.images.remove(img)

# --- 4. Camera ray directions per pixel ---
cam = scene.camera
cam_mtx = cam.matrix_world
# Blender camera: view_frame gives the 4 corners at the near plane in cam space.
frame = [cam_mtx @ v for v in cam.data.view_frame(scene=scene)]
# frame order: top-right, bottom-right, bottom-left, top-left (cam looks -Z)
tr, br, bl, tl = frame
origin = cam_mtx.translation


def ray_dir(px, py):
    # px,py in [0,1], py=0 top
    top = tl.lerp(tr, px)
    bot = bl.lerp(br, px)
    p = bot.lerp(top, 1 - py)
    d = (p - origin).normalized()
    return np.array([d.x, d.y, d.z], dtype=np.float32)


# --- 5. Compare on a coarse grid of pixels ---
print("\npx,py    native(0.5*sky)      vibrt_sampled(0.5*sky)   ratio")
diffs = []
for py in (0.1, 0.25, 0.4, 0.5, 0.6, 0.75):
    for px in (0.25, 0.5, 0.75):
        d = ray_dir(px, py)
        nat = native[int(py * (rh - 1)), int(px * (rw - 1))]
        smp = sample_equirect(sky_rgb, d) * STRENGTH_A
        r = smp / np.maximum(nat, 1e-6)
        diffs.append((nat, smp))
        print(f"({px:.2f},{py:.2f}) {np.round(nat,4)}  {np.round(smp,4)}  {np.round(r,3)}")

nat_all = np.array([d[0] for d in diffs])
smp_all = np.array([d[1] for d in diffs])
print(f"\nmean native = {np.round(nat_all.mean(0),4)}")
print(f"mean vibrt  = {np.round(smp_all.mean(0),4)}")
print(f"overall ratio = {np.round(smp_all.mean(0)/np.maximum(nat_all.mean(0),1e-6),3)}")
print("=== DONE ===", flush=True)
