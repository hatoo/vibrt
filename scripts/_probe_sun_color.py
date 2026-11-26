"""Compare peak-pixel color to flux-weighted sun-split color.

Hypothesis: the split picks up too much warm corona, so the
"flux-weighted" RGB the device sees as the sun is yellower than the
actual disc center. If `peak` is significantly cooler than `mean`, the
delta sun light is biasing every directly-lit surface warm.
"""
import os, sys, math
import numpy as np
import bpy

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))

import vibrt_blender.exporter as exporter
from vibrt_blender.export_context import ExportContext
export_context = ExportContext()

# Replace the sun extractor with a passthrough that records the raw bake
# pre-clamp, so we can sample the actual disc-center pixel.
orig_extract = exporter._extract_sun_from_bake_inplace
captured = {}

def _spy(rgb, w, h, sky_node, world_name):
    captured["rgb_pre"] = rgb.copy()
    return orig_extract(rgb, w, h, sky_node, world_name)

exporter._extract_sun_from_bake_inplace = _spy

scene = bpy.context.scene
exporter.clear_sky_bake_cache(context=export_context)
exporter.prebake_sky_envmaps_for_world(scene.world, context=export_context)

rgb_pre = captured["rgb_pre"]
h, w, _ = rgb_pre.shape
lum = 0.2126*rgb_pre[..., 0] + 0.7152*rgb_pre[..., 1] + 0.0722*rgb_pre[..., 2]
peak_idx = np.unravel_index(np.argmax(lum), lum.shape)
peak_rgb = rgb_pre[peak_idx]
peak_lum = lum[peak_idx]
print(f"peak pixel @ ({peak_idx[1]},{peak_idx[0]}): rgb=({peak_rgb[0]:.1f}, {peak_rgb[1]:.1f}, {peak_rgb[2]:.1f}) lum={peak_lum:.1f}")
peak_max = max(float(peak_rgb[0]), float(peak_rgb[1]), float(peak_rgb[2]))
print(f"  normalised: ({peak_rgb[0]/peak_max:.4f}, {peak_rgb[1]/peak_max:.4f}, {peak_rgb[2]/peak_max:.4f})")
print(f"  R/G = {peak_rgb[0]/max(peak_rgb[1],1):.4f}, B/G = {peak_rgb[2]/max(peak_rgb[1],1):.4f}")

# Top 3 / 10 / 30 pixel averages
for k in (3, 10, 30, 100):
    flat_idx = np.argpartition(lum.ravel(), -k)[-k:]
    rgbs = rgb_pre.reshape(-1, 3)[flat_idx]
    lums = lum.ravel()[flat_idx]
    # luminance-weighted average
    w_avg = (rgbs * lums[:, None]).sum(axis=0) / lums.sum()
    m = max(w_avg)
    print(f"\ntop {k} pixels (lum-weighted avg) lum_min={lums.min():.1f}, lum_max={lums.max():.1f}")
    print(f"  rgb_avg = ({w_avg[0]:.1f}, {w_avg[1]:.1f}, {w_avg[2]:.1f})")
    print(f"  normalised: ({w_avg[0]/m:.4f}, {w_avg[1]/m:.4f}, {w_avg[2]/m:.4f})")
    print(f"  R/G = {w_avg[0]/max(w_avg[1],1):.4f}, B/G = {w_avg[2]/max(w_avg[1],1):.4f}")

# Current threshold-based split (what the engine actually uses):
print(f"\n=== current split (threshold = max(30, p99*3)) ===")
print(f"  threshold-extracted pixels above implicit cap; sun light color was already")
print(f"  printed by the engine — re-extracting via the cached split:")
key = exporter._sky_node_cache_key(scene.world,
    next(n for n in scene.world.node_tree.nodes if n.bl_idname == "ShaderNodeTexSky"))
baked = export_context.sky_maps.get(key)
if baked is not None:
    _, _, _, sun_light = baked
    if sun_light is not None:
        print(f"  sun.color   = {sun_light['color']}")
        print(f"  sun.strength = {sun_light['strength']:.2f}")
        c = sun_light['color']
        print(f"  R/G = {c[0]/max(c[1],1e-9):.4f}, B/G = {c[2]/max(c[1],1e-9):.4f}")
