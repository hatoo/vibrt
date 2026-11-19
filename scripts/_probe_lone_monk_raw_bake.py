"""Look at the RAW bake peak (pre sun extraction) to see if Cycles' bake
saturates somewhere or if our extractor is over-clipping."""
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

scene = bpy.context.scene
# Bake without using the cache so we can read the raw bake
exporter._SKY_BAKE_CACHE.clear()
world = scene.world
sky_node = exporter._world_sky_node(world)
print(f"sky_node = {sky_node}")
print(f"sky_node sun_disc = {sky_node.sun_disc}")
print(f"sky_node sun_size (rad) = {sky_node.sun_size}")
print(f"sky_node sun_intensity = {sky_node.sun_intensity}")
print(f"sky_node sun_elevation (rad) = {sky_node.sun_elevation}")
print(f"sky_node sun_rotation (rad) = {sky_node.sun_rotation}")
print(f"sky_node turbidity = {sky_node.turbidity}")

# Skip the cached prebake; do a raw bake at higher res
rgb = exporter._bake_sky_world_to_pixels(world, w=2048, h=1024)
if rgb is None:
    print("bake failed!")
    raise SystemExit
arr, w, h = rgb
print(f"\n=== RAW bake (pre-extract) ===")
print(f"  shape = {arr.shape}, dtype = {arr.dtype}")
lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
print(f"  R: mean={arr[..., 0].mean():.4f}, peak={arr[..., 0].max():.4f}")
print(f"  G: mean={arr[..., 1].mean():.4f}, peak={arr[..., 1].max():.4f}")
print(f"  B: mean={arr[..., 2].mean():.4f}, peak={arr[..., 2].max():.4f}")
print(f"  luminance: mean={lum.mean():.4f}, peak={lum.max():.4f}")
yi, xi = np.unravel_index(lum.argmax(), lum.shape)
print(f"  brightest pixel @ ({yi}, {xi}): R={arr[yi, xi, 0]:.3f}, G={arr[yi, xi, 1]:.3f}, B={arr[yi, xi, 2]:.3f}")
# Histogram
percentiles = [50, 75, 90, 95, 99, 99.9, 99.99, 99.999, 100]
print(f"\n  luminance percentiles:")
for p in percentiles:
    v = np.percentile(lum, p)
    print(f"    p{p:>7}: {v:.4f}")

# How much energy is in the top 0.1% pixels (sun cone area)?
threshold = np.percentile(lum, 99.9)
mask = lum >= threshold
sun_flux = arr[mask].sum(axis=0) * (4 * np.pi / (h * w))  # total flux through 4π solid angle
print(f"\n  flux in top 0.1% pixels (rough sun cone): R={sun_flux[0]:.2f}, G={sun_flux[1]:.2f}, B={sun_flux[2]:.2f}")
print(f"    ({mask.sum()} pixels above lum={threshold:.3f})")
