"""Run pabellon's Sky-alone prebake and dump the resulting equirect to
PNG so we can inspect whether PREETHAM at sun_elevation=90° actually
produces a hot zenith sun (or whether something in our bake setup
turns it into a cool diffuse blue dome)."""
import bpy, os, sys, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

from vibrt_blender.exporter import (
    prebake_sky_envmaps_for_world, _SKY_BAKE_CACHE,
)

prebake_sky_envmaps_for_world(bpy.context.scene.world, w=1024, h=512)

print("\n=== _SKY_BAKE_CACHE entries ===")
for key, (rgb_bytes, w, h, sun) in _SKY_BAKE_CACHE.items():
    if isinstance(rgb_bytes, np.ndarray):
        rgb = rgb_bytes
    else:
        rgb = np.frombuffer(rgb_bytes, dtype=np.float32)
    rgb = rgb.reshape((h, w, 3))
    print(f"  key={key!r}")
    print(f"  size={w}x{h}  mean={tuple(round(float(x),3) for x in rgb.mean(axis=(0,1)))}  max={float(rgb.max()):.2f}  median={float(np.median(rgb)):.4f}")
    lum = 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]
    print(f"  luminance: max={float(lum.max()):.2f}  p50={float(np.percentile(lum, 50)):.4f}  p99={float(np.percentile(lum, 99)):.4f}")
    print(f"  ratio max/median = {float(lum.max() / max(np.median(lum), 1e-4)):.2f}")
    print(f"  top half (zenith) mean: {tuple(round(float(x),3) for x in rgb[:h//2].mean(axis=(0,1)))}")
    print(f"  bottom half mean: {tuple(round(float(x),3) for x in rgb[h//2:].mean(axis=(0,1)))}")
    yy, xx = np.unravel_index(np.argmax(lum), lum.shape)
    print(f"  brightest pixel at (x,y) = ({xx},{yy}) → uv ({xx/w:.3f}, {yy/h:.3f}) value {tuple(round(float(c),2) for c in rgb[yy,xx])}")
    print(f"  sun light: {sun}")

    # save as PNG for visual inspection
    out_path = os.path.join(HERE, "test_scenes", "pabellon", "_sky_bake_inspect.png")
    rgb_clip = np.clip(rgb / (1 + rgb), 0, 1)  # Reinhard
    rgb_srgb = np.where(rgb_clip <= 0.0031308, rgb_clip * 12.92,
                         1.055 * np.power(rgb_clip, 1/2.4) - 0.055)
    out_arr = np.dstack([np.clip(rgb_srgb, 0, 1),
                         np.ones((h, w), dtype=np.float32)]).astype(np.float32)
    out_arr = out_arr[::-1].reshape(-1)
    img = bpy.data.images.new("_skybake_dump", w, h, alpha=False)
    img.pixels.foreach_set(out_arr)
    img.filepath_raw = out_path
    img.file_format = "PNG"
    img.save()
    print(f"  wrote {out_path}")
