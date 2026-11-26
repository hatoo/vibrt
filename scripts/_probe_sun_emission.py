"""Inspect the sun light extracted from the bake — direction & RGB
emission — so we can correlate with the pool floor brightness."""
import bpy, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender.exporter as exporter
from vibrt_blender.export_context import ExportContext
export_context = ExportContext()

scene = bpy.context.scene
world = scene.world

import time
export_context.sky_maps.clear()
exporter.prebake_sky_envmaps_for_world(world, context=export_context)

key = exporter._world_generic_cache_key(world) if exporter._world_needs_full_bake(world) else exporter._sky_node_cache_key(world, exporter._world_sky_node(world))
cached = export_context.sky_maps[key]
rgb, w, h, sun = cached
print(f"bake: {w}x{h}, sun extracted: {sun is not None}")
if sun is not None:
    print(f"  direction: {sun.get('direction', 'n/a')}")
    print(f"  emission: {sun.get('emission', 'n/a')}")
    print(f"  cos_angle: {sun.get('cos_angle', 'n/a')}")

# Sky bake stats
import numpy as np
print(f"\nsky bake stats:")
print(f"  mean rgb: {tuple(round(float(x),3) for x in rgb.mean(axis=(0,1)))}")
print(f"  max rgb:  {tuple(round(float(x),3) for x in rgb.max(axis=(0,1)))}")
print(f"  99th %ile per ch: {tuple(round(float(np.percentile(rgb[..., c], 99)),3) for c in range(3))}")
