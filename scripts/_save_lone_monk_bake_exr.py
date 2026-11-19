"""Bake lone_monk's Nishita Sky Texture into a 2048×1024 EXR at a
fixed path (`c:/tmp/lone_monk_nishita_bake.exr`) for later swap-in.
Also bake an *unedited* version (no sun extraction) so we can test
with and without the residual-sky sun split.
"""
import bpy
import sys
import os
import math
import numpy as np

sys.path.insert(0, r"C:\Users\hato2\Desktop\vibrt\blender")
from vibrt_blender import exporter

w = bpy.context.scene.world
exporter.clear_sky_bake_cache()

# Run vibrt's bake (this will populate the cache and *also* extract
# the sun, mutating the rgb in place).
exporter.prebake_sky_envmaps_for_world(w, w=2048, h=1024)
key = next(iter(exporter._SKY_BAKE_CACHE.keys()))
rgb, bw, bh, sun = exporter._SKY_BAKE_CACHE[key]
arr = np.asarray(rgb).reshape(bh, bw, 3).astype(np.float32)
print(f"after sun extraction: max L = {arr.max():.4f}  sun = {sun}")

# Use Blender's image API to save as EXR.
def save_exr(path, arr, bw, bh):
    img = bpy.data.images.new("__tmp_save", width=int(bw), height=int(bh),
                               alpha=False, float_buffer=True,
                               is_data=True)
    rgba = np.zeros((bh, bw, 4), dtype=np.float32)
    rgba[..., :3] = arr
    rgba[..., 3] = 1.0
    img.pixels.foreach_set(rgba.flatten().tolist())
    img.filepath_raw = path
    img.file_format = "OPEN_EXR"
    img.save()
    bpy.data.images.remove(img)

residual_path = r"c:/tmp/lone_monk_nishita_residual.exr"
save_exr(residual_path, arr, bw, bh)
print(f"wrote {residual_path}")

# Now bake again WITHOUT sun extraction so we can test the full envmap.
exporter.clear_sky_bake_cache()
# Temporarily monkey-patch the extractor to skip its work.
orig_extract = exporter._extract_sun_from_bake_inplace
exporter._extract_sun_from_bake_inplace = lambda *a, **kw: None
try:
    exporter.prebake_sky_envmaps_for_world(w, w=2048, h=1024)
finally:
    exporter._extract_sun_from_bake_inplace = orig_extract

key = next(iter(exporter._SKY_BAKE_CACHE.keys()))
rgb_full, bw, bh, _ = exporter._SKY_BAKE_CACHE[key]
arr_full = np.asarray(rgb_full).reshape(bh, bw, 3).astype(np.float32)
print(f"\nFULL bake (no sun extraction): max L = {arr_full.max():.4f}, "
      f"mean L = {arr_full.mean():.4f}")

full_path = r"c:/tmp/lone_monk_nishita_full.exr"
save_exr(full_path, arr_full, bw, bh)
print(f"wrote {full_path}")
