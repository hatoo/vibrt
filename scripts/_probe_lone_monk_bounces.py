"""Bounce-by-bounce probe for lone_monk to localise where vibrt under-
delivers light. Tests dark / mid / bright regions separately so we can
tell whether the gap is in direct lighting (NEE), 1-bounce indirect, or
deeper indirect."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "blender"))

import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass


def render(engine, spp, db, out_path, percentage=50):
    scene = bpy.context.scene
    scene.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        scene.vibrt_spp = spp
    else:
        scene.cycles.samples = spp
        scene.cycles.use_adaptive_sampling = False
    scene.cycles.diffuse_bounces = db
    scene.render.resolution_percentage = percentage
    scene.render.image_settings.file_format = "PNG"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


def region_stats(path):
    img = bpy.data.images.load(str(path))
    img.colorspace_settings.name = "sRGB"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    arr = flat.reshape((h, w, 4))[..., :3]
    # Compute brightness buckets — lone_monk's gap is dominated by dark regions
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    p25 = np.percentile(lum, 25)
    p75 = np.percentile(lum, 75)
    mask_dark = lum < p25
    mask_mid = (lum >= p25) & (lum < p75)
    mask_bright = lum >= p75
    out = {}
    for label, mask in (("all", np.ones_like(mask_dark)),
                         ("dark25", mask_dark),
                         ("mid50", mask_mid),
                         ("bright25", mask_bright)):
        if mask.sum() > 0:
            mean = arr[mask].mean(axis=0)
            out[label] = tuple(round(float(c), 4) for c in mean)
        else:
            out[label] = (0, 0, 0)
    bpy.data.images.remove(img)
    return out


def main():
    spp = 128
    pct = 50
    rows = []
    for engine in ("vibrt", "cycles"):
        for db in (0, 1, 2):
            out = HERE / "test_scenes" / "lone_monk" / f"_probe_{engine}_db{db}.png"
            print(f"\n=== {engine} db={db} ===", flush=True)
            render(engine, spp, db, out, pct)
            stats = region_stats(out)
            rows.append((engine, db, stats))
            for k, v in stats.items():
                print(f"  {k}: {v}", flush=True)
    print("\n=== summary (mean RGB) ===")
    for engine, db, stats in rows:
        print(f"  {engine:6s} db={db}: all={stats['all']} dark25={stats['dark25']} bright25={stats['bright25']}")


main()
