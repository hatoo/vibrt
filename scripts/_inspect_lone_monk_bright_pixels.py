"""Inspect where the brightness gap between vibrt and cycles is concentrated
on the full lone_monk db=0 EXRs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import bpy
import numpy as np

HERE = Path(__file__).resolve().parent.parent

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
    v = load_exr(HERE / "test_scenes" / "lone_monk" / "_dbsweep_db0_vibrt.exr")
    c = load_exr(HERE / "test_scenes" / "lone_monk" / "_dbsweep_db0_cycles.exr")
    H, W = min(v.shape[0], c.shape[0]), min(v.shape[1], c.shape[1])
    v, c = v[:H, :W], c[:H, :W]
    lum_v = (v * np.array([0.2126, 0.7152, 0.0722])).sum(axis=-1)
    lum_c = (c * np.array([0.2126, 0.7152, 0.0722])).sum(axis=-1)

    print(f"image: {H}x{W}")
    print(f"vibrt total energy : {v.sum():.0f}")
    print(f"cycles total energy: {c.sum():.0f}")
    print(f"surplus (v - c)    : {(v - c).sum():.0f}  ({(v - c).sum() / c.sum() * 100:.1f}% of cycles)")

    # bucket by cycles luminance
    buckets = [(0, 0.01, "dark"), (0.01, 0.1, "mid_low"), (0.1, 1, "mid"),
               (1, 5, "bright"), (5, 50, "super"), (50, 1e9, "ultra")]
    print(f"\n{'bucket':>10} {'n':>8} {'sum_v':>10} {'sum_c':>10} {'ratio v/c':>10} {'surplus':>12}")
    for lo, hi, name in buckets:
        mask = (lum_c >= lo) & (lum_c < hi)
        if not mask.any():
            print(f"{name:>10} {0:>8d}")
            continue
        n = int(mask.sum())
        sv = float(v[mask].sum())
        sc = float(c[mask].sum())
        r = sv / max(sc, 1e-6)
        s = sv - sc
        print(f"{name:>10} {n:>8d} {sv:>10.1f} {sc:>10.1f} {r:>10.3f} {s:>+12.1f}")

    # Top 20 brightest vibrt pixels — where are they, what's cycles say?
    print("\n=== top 20 vibrt-luminance pixels: vs cycles ===")
    flat_idx = np.argsort(lum_v.ravel())[::-1][:20]
    for i, idx in enumerate(flat_idx):
        y, x = divmod(int(idx), W)
        print(f"  #{i:>2}: y={y:>3} x={x:>3}   vibrt={tuple(round(float(v[y, x, k]), 2) for k in range(3))}  "
              f"cycles={tuple(round(float(c[y, x, k]), 2) for k in range(3))}  "
              f"v/c={lum_v[y, x] / max(lum_c[y, x], 1e-6):.2f}")

    # Top 20 brightest cycles pixels
    print("\n=== top 20 cycles-luminance pixels: vs vibrt ===")
    flat_idx = np.argsort(lum_c.ravel())[::-1][:20]
    for i, idx in enumerate(flat_idx):
        y, x = divmod(int(idx), W)
        print(f"  #{i:>2}: y={y:>3} x={x:>3}   vibrt={tuple(round(float(v[y, x, k]), 2) for k in range(3))}  "
              f"cycles={tuple(round(float(c[y, x, k]), 2) for k in range(3))}  "
              f"v/c={lum_v[y, x] / max(lum_c[y, x], 1e-6):.2f}")


main()
