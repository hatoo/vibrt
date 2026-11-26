"""Inspect lone_monk's sky bake to see if sun extraction worked properly."""
import os
import sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))

import bpy
import numpy as np

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
    print(f"    shape={rgb.shape}, peak={rgb.max():.3f}, mean={rgb.mean():.3f}")
    print(f"    sun={sun}")

# Look at the bake stats
if cache:
    rgb, w, h, sun = next(iter(cache.values()))
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    print(f"\n=== bake luminance buckets ===")
    p = [50, 75, 90, 95, 99, 99.5, 99.9, 100]
    for q in p:
        v = np.percentile(lum, q)
        print(f"  p{q:>4}: {v:.3f}")
    # Per-channel mean / peak
    print(f"\n  R mean={rgb[..., 0].mean():.3f}, peak={rgb[..., 0].max():.3f}")
    print(f"  G mean={rgb[..., 1].mean():.3f}, peak={rgb[..., 1].max():.3f}")
    print(f"  B mean={rgb[..., 2].mean():.3f}, peak={rgb[..., 2].max():.3f}")

    # Per-row mean (zenith=top, horizon=middle, nadir=bottom)
    print(f"\n=== row breakdown ===")
    print(f"  top row 0          (zenith): "
          f"({rgb[0, :, 0].mean():.3f}, {rgb[0, :, 1].mean():.3f}, {rgb[0, :, 2].mean():.3f})")
    print(f"  row {h//8} (~22.5° from zenith): "
          f"({rgb[h//8, :, 0].mean():.3f}, {rgb[h//8, :, 1].mean():.3f}, {rgb[h//8, :, 2].mean():.3f})")
    print(f"  row {h//4} (~45°): "
          f"({rgb[h//4, :, 0].mean():.3f}, {rgb[h//4, :, 1].mean():.3f}, {rgb[h//4, :, 2].mean():.3f})")
    print(f"  row {h//2} (horizon): "
          f"({rgb[h//2, :, 0].mean():.3f}, {rgb[h//2, :, 1].mean():.3f}, {rgb[h//2, :, 2].mean():.3f})")
    # Find the brightest pixel
    yi, xi = np.unravel_index(lum.argmax(), lum.shape)
    print(f"\n  brightest pixel @ ({yi}, {xi}) (h={h}, w={w}): "
          f"R={rgb[yi, xi, 0]:.3f}, G={rgb[yi, xi, 1]:.3f}, B={rgb[yi, xi, 2]:.3f}, "
          f"lum={lum[yi, xi]:.3f}")
    # Theta from zenith for that pixel
    theta_deg = (yi + 0.5) / h * 180.0
    print(f"  theta from zenith ≈ {theta_deg:.1f}° (sun_elevation should map to "
          f"theta = 90° − elev_deg)")
