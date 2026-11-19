"""Investigate where classroom mid-wall under-brightness comes from.

Renders classroom at increasing diffuse_bounces with both vibrt and
Cycles, then reports mid-wall mean RGB. If the gap is already large at
diffuse=0 the issue is NEE (direct light delivery). If the gap grows
with bounce count, it's indirect-light propagation.

Run via:
  blender -b test_scenes/classroom/classroom/classroom.blend \
      --python scripts/_probe_classroom_bounces.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "blender"))

# Force-enable the working-tree addon so VIBRT renders pick up the
# current devicecode.cu / material_export.py.
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass


def render(engine: str, spp: int, db: int, out_path: Path, percentage: int = 50):
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


def midwall_stats(path: Path):
    img = bpy.data.images.load(str(path))
    img.colorspace_settings.name = "sRGB"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    arr = flat.reshape((h, w, 4))[..., :3]
    # `midwall_stats` follows _compare_classroom_regions.py: bottom
    # 30-50% of the image (Blender pixels are origin-bottom-left, so
    # rows from h*0.50 to h*0.70 in foreach_get's order = visual top
    # 30-50%). Empirically the largest gap row band.
    y0 = int(h * 0.50)
    y1 = int(h * 0.70)
    region = arr[y0:y1, :, :]
    mean = region.mean(axis=(0, 1))
    bpy.data.images.remove(img)
    return tuple(round(float(c), 4) for c in mean)


def main():
    spp = 64  # tighter to manage time
    pct = 50
    rows = []
    for engine in ("vibrt", "cycles"):
        for db in (0, 1, 2):
            out = HERE / "test_scenes" / "classroom" / f"_probe_{engine}_db{db}.png"
            print(f"\n=== {engine} diffuse_bounces={db} ===", flush=True)
            render(engine, spp, db, out, pct)
            mean = midwall_stats(out)
            rows.append((engine, db, mean))
            print(f"  midwall mean RGB = {mean}", flush=True)
    print("\n=== summary ===")
    for engine, db, mean in rows:
        print(f"  {engine:6s} db={db}: {mean}")


main()
