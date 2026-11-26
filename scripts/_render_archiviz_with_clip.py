"""Render flat-archiviz with cam_clip_start overridden so we can see whether the
remaining left-corner black is caused by a piece of geometry that sits between
2m and some larger distance from the camera."""
import bpy, sys, os, json
sys.path.insert(0, os.path.abspath("blender"))
import vibrt_blender.exporter as exporter
from vibrt_blender.export_context import ExportContext
export_context = ExportContext()
from vibrt_blender import runner

# Pre-bake the world (engine.update() does this normally).
exporter.clear_sky_bake_cache(context=export_context)
exporter.prebake_sky_envmaps_for_world(bpy.context.scene.world, context=export_context)

# Force a different clip_start.
override_clip = float(os.environ.get("CLIPOVR", "5.0"))
out_path = os.environ.get("OUTPNG", "test_scenes/flat_archiviz/_clip_probe.png")
print(f"override clip_start = {override_clip}m  → {out_path}")

orig_export_camera = exporter._export_camera
def patched_export_camera(scene, cam_obj, aspect):
    blob = orig_export_camera(scene, cam_obj, aspect)
    blob["clip_start"] = override_clip
    return blob
exporter._export_camera = patched_export_camera

depsgraph = bpy.context.evaluated_depsgraph_get()
scene_json_str, scene_bin, tex_arrays = exporter.export_scene_to_memory(depsgraph, context=export_context)

def _report(level, msg):
    print(f"[{level}] {msg}")
def _is_break():
    return False

pixels, depth = runner.run_render_inproc(
    scene_json_str, scene_bin, _report, _is_break,
    denoise=False, texture_arrays=tex_arrays,
)

# Save via numpy/PIL
import numpy as np
H, W = pixels.shape[:2]
arr = np.asarray(pixels[..., :3], dtype=np.float32)

# Dump diagnostic at the corner pixels we care about.
# pixels comes back from run_render_inproc as (H, W, 4) with row 0 == top
# (it has already been flipped vs the kernel's bottom-up internal buffer
# inside run_render_inproc -> see runner.py).
print(f"\n=== diag: pixels[y, x] (linear RGB) and depth[y, x] (m) ===")
print(f"shape: pixels={arr.shape}  depth={depth.shape}")
for (label, y, x) in [
    ("TL_corner",  0,    0),
    ("TL_inner",   60,   60),
    ("TL_arc",     0,    150),
    ("ML_edge",    H//2, 0),
    ("ML_inner",   H//2, 60),
    ("BL_corner",  H-1,  0),
    ("BL_inner",   H-61, 60),
    ("center",     H//2, W//2),
    ("TR_corner",  0,    W-1),
    ("BR_corner",  H-1,  W-1),
]:
    px = arr[y, x]
    d = float(depth[y, x])
    is_miss = d > 999.0  # clip_end is 1000
    tag = "MISS" if is_miss else "HIT "
    print(f"  {label:12s} y={y:4d} x={x:4d}  rgb=({px[0]:.4f}, {px[1]:.4f}, {px[2]:.4f})  d={d:8.3f}m  {tag}")

arr = np.clip(arr, 0.0, 1.0)
def linear_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92*c, 1.055*np.power(c, 1/2.4) - 0.055)
srgb = linear_to_srgb(arr)
# Use Blender's bpy.data.images to save (always available inside bpy).
flat_rgba = np.empty((H, W, 4), dtype=np.float32)
flat_rgba[..., :3] = srgb
flat_rgba[..., 3] = 1.0
img_name = f"__vibrt_probe_{os.path.basename(out_path)}"
if img_name in bpy.data.images:
    bpy.data.images.remove(bpy.data.images[img_name])
img = bpy.data.images.new(img_name, W, H, alpha=True, float_buffer=False)
img.pixels.foreach_set(np.flipud(flat_rgba).ravel())  # foreach wants bottom-up
img.filepath_raw = os.path.abspath(out_path)
img.file_format = 'PNG'
img.save()
print(f"saved {os.path.abspath(out_path)}")
