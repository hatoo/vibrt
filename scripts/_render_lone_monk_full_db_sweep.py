"""Full lone_monk scene, compositor=OFF, view=Raw, sweep diffuse_bounces
to localize where vibrt's 66% over-delivery vs cycles originates.

db=0 → direct NEE only (sun + envmap + mesh-light)
db=1,2 → +1, +2 indirect diffuse bounces

If db=0 matches and gap grows with db → indirect bounce path issue.
If db=0 already over-delivers → direct NEE issue (mesh lights? sun? envmap?).
"""
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


def configure(engine: str, spp: int, percentage: int, db: int):
    scene = bpy.context.scene
    scene.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        scene.vibrt_spp = spp
    else:
        scene.cycles.samples = spp
        scene.cycles.use_adaptive_sampling = False
    scene.cycles.diffuse_bounces = db
    scene.render.resolution_percentage = percentage
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "32"
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.use_compositing = False


def render_to(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


def load_exr(path: Path):
    img = bpy.data.images.load(str(path))
    img.colorspace_settings.name = "Linear Rec.709"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    arr = flat.reshape((h, w, 4))[..., :3]
    bpy.data.images.remove(img)
    return arr


def main():
    spp = int(os.environ.get("SPP", "64"))
    pct = int(os.environ.get("PCT", "25"))
    print(f"\n=== full lone_monk db sweep @ spp={spp}, pct={pct}, compositor=OFF ===\n", flush=True)

    rows = []
    for db in (0, 1, 2):
        out_v = HERE / "test_scenes" / "lone_monk" / f"_dbsweep_db{db}_vibrt.exr"
        out_c = HERE / "test_scenes" / "lone_monk" / f"_dbsweep_db{db}_cycles.exr"
        print(f"--- db={db} vibrt ---", flush=True)
        configure("vibrt", spp, pct, db)
        render_to(out_v)
        print(f"--- db={db} cycles ---", flush=True)
        configure("cycles", spp, pct, db)
        render_to(out_c)
        v = load_exr(out_v)
        c = load_exr(out_c)
        if v.shape != c.shape:
            h = min(v.shape[0], c.shape[0])
            w = min(v.shape[1], c.shape[1])
            v = v[:h, :w]
            c = c[:h, :w]
        ratio = float(v.sum() / max(c.sum(), 1e-6))
        rows.append((db, v.mean(axis=(0, 1)), c.mean(axis=(0, 1)), ratio,
                     float(v.max()), float(c.max())))

    print(f"\n=== db sweep results (compositor=OFF, view=Raw) ===")
    print(f"{'db':>3} | {'vibrt mean RGB':>30} | {'cycles mean RGB':>30} | {'ratio v/c':>10} | {'v peak':>10} | {'c peak':>10}")
    for db, vm, cm, r, vp, cp in rows:
        print(f"{db:>3} | "
              f"({vm[0]:6.3f}, {vm[1]:6.3f}, {vm[2]:6.3f}) | "
              f"({cm[0]:6.3f}, {cm[1]:6.3f}, {cm[2]:6.3f}) | "
              f"{r:>10.3f} | {vp:>10.2f} | {cp:>10.2f}")


main()
