"""Run the same world-bake the addon performs and save the equirect to
PNG so we can see why pabellon's sky reads as 'rough'.

Usage:
  blender -b test_scenes/pabellon/3d/pavillon_barcelone_v1.2.blend --python scripts/_dump_pabellon_skybake.py
"""
import os, sys, math
import bpy
import numpy as np

# Force-load the working-tree addon (mirrors what scripts/_blender_render.py does).
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender.exporter as exporter

scene = bpy.context.scene
world = scene.world
print(f"world: {world.name}")
print(f"  full bake needed? {exporter._world_needs_full_bake(world)}")
print(f"  has sky node? {exporter._world_sky_node(world) is not None}")

# Try multiple resolutions and report bake time + visible roughness proxies.
for (w, h) in [(1024, 512), (2048, 1024), (4096, 2048)]:
    exporter._SKY_BAKE_CACHE.clear()
    import time
    t0 = time.perf_counter()
    exporter.prebake_sky_envmaps_for_world(world, w=w, h=h)
    dt = time.perf_counter() - t0
    key = exporter._world_generic_cache_key(world) if exporter._world_needs_full_bake(world) else exporter._sky_node_cache_key(world, exporter._world_sky_node(world))
    cached = exporter._SKY_BAKE_CACHE.get(key)
    if cached is None:
        print(f"  {w}x{h}: bake failed")
        continue
    rgb, bw, bh, sun = cached
    # Save raw RGB linear EXR-ish via PNG (clipped to [0,1]).
    out = os.path.join(HERE, "test_scenes", "pabellon", f"_skybake_{bw}x{bh}.png")
    img = bpy.data.images.new(f"sb_{bw}x{bh}", width=bw, height=bh, alpha=False, float_buffer=True)
    rgba = np.concatenate([rgb, np.ones((bh, bw, 1), dtype=np.float32)], axis=-1)
    img.pixels.foreach_set(rgba.reshape(-1))
    img.filepath_raw = out
    img.file_format = "PNG"
    # Apply scene view settings to the saved file (no need to override).
    sc = bpy.context.scene
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.exposure = 0.0
    sc.render.image_settings.file_format = "PNG"
    img.save_render(filepath=out)
    # Stats: per-channel mean, max, std along horizon strip (y around the equator).
    y0 = bh // 3
    y1 = 2 * bh // 3
    horizon = rgb[y0:y1, :, :]
    print(f"  {w}x{h} ({dt:.2f}s) -> mean={tuple(round(float(x),3) for x in horizon.mean(axis=(0,1)))} max={float(rgb.max()):.3f} std={float(horizon.std()):.3f} -> {out}")
