"""Inspect the portal bake's pixel stats (mean / peak / sun pixel
brightness) and compare to Cycles' direct TexSky eval at the same
direction. Helps diagnose the right/floor +0.06 over-bright in the
post-fix classroom render."""
import os
import sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))

import bpy
import numpy as np
import math

import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

from vibrt_blender import exporter
from vibrt_blender.export_context import ExportContext
export_context = ExportContext()

scene = bpy.context.scene
exporter.clear_sky_bake_cache(context=export_context)
exporter.prebake_sky_envmaps_for_world(scene.world, context=export_context)
exporter.prebake_portal_emissions_for_scene(scene, context=export_context)

cache = export_context.sky_maps
print(f"=== export_context.sky_maps keys ===")
for k in cache.keys():
    rgb, w, h, sun = cache[k]
    print(f"  {k!r}")
    print(f"    shape={rgb.shape}, peak={rgb.max():.3f}, mean={rgb.mean():.3f}, "
          f"sun={sun}")

# Bake stats
portal_keys = [k for k in cache.keys() if k.startswith("__portal_emit__")]
if portal_keys:
    rgb, bw, bh, sun = cache[portal_keys[0]]
    print(f"\n=== portal bake ({portal_keys[0]}) ===")
    print(f"  shape={rgb.shape}, dtype={rgb.dtype}")
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    print(f"  mean RGB = ({rgb[..., 0].mean():.3f}, {rgb[..., 1].mean():.3f}, {rgb[..., 2].mean():.3f})")
    print(f"  peak RGB at any pixel = ({rgb[..., 0].max():.3f}, {rgb[..., 1].max():.3f}, {rgb[..., 2].max():.3f})")
    print(f"  luminance mean = {lum.mean():.3f}")
    print(f"  luminance peak = {lum.max():.3f}, at pixel idx = {np.unravel_index(lum.argmax(), lum.shape)}")
    # Sun is at zenith → top row (small v, large u). Pixel grid: row 0 = north pole?
    # _bake_sky_world_to_pixels uses Cycles PANO/EQUIRECTANGULAR, conventional
    # equirect: row 0 = top (zenith if camera is +Y up, which is the
    # default).
    print(f"  top-row mean (zenith) = "
          f"({rgb[0, :, 0].mean():.3f}, {rgb[0, :, 1].mean():.3f}, {rgb[0, :, 2].mean():.3f})")
    print(f"  bottom-row mean (nadir) = "
          f"({rgb[-1, :, 0].mean():.3f}, {rgb[-1, :, 1].mean():.3f}, {rgb[-1, :, 2].mean():.3f})")
    print(f"  middle-row mean (horizon) = "
          f"({rgb[bh//2, :, 0].mean():.3f}, {rgb[bh//2, :, 1].mean():.3f}, {rgb[bh//2, :, 2].mean():.3f})")
    # Histogram of luminance
    bins = [0, 1, 5, 10, 50, 100, 1e9]
    hist, _ = np.histogram(lum, bins=bins)
    total = lum.size
    print(f"  luminance histogram (% of pixels):")
    for i, n in enumerate(hist):
        pct = 100 * n / total
        print(f"    {bins[i]:>6.1f} - {bins[i+1]:>6.1f}: {pct:5.2f}%  ({n} pixels)")
