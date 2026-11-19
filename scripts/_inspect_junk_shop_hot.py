"""Find hot pixels in junk_shop EXR comparison and identify their source."""
from __future__ import annotations
from pathlib import Path
import bpy
import numpy as np

HERE = Path(__file__).resolve().parent.parent

def load(p):
    img = bpy.data.images.load(str(p))
    img.colorspace_settings.name = "Linear Rec.709"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    arr = flat.reshape((h, w, 4))[..., :3]
    bpy.data.images.remove(img)
    return arr


v = load(HERE / "test_scenes" / "_eng_diff" / "junk_shop_vibrt.exr")
c = load(HERE / "test_scenes" / "_eng_diff" / "junk_shop_cycles.exr")

H, W = v.shape[:2]
print(f"shape: {H} x {W}")
print(f"vibrt total : {v.sum():.0f}")
print(f"cycles total: {c.sum():.0f}")
print(f"ratio v/c   : {v.sum() / max(c.sum(), 1e-6):.3f}")

lum_v = (v * np.array([0.2126, 0.7152, 0.0722])).sum(axis=-1)
lum_c = (c * np.array([0.2126, 0.7152, 0.0722])).sum(axis=-1)

# Bucket
print(f"\n  vibrt lum percentiles: p50={np.percentile(lum_v, 50):.3f}  "
      f"p99={np.percentile(lum_v, 99):.3f}  "
      f"p99.9={np.percentile(lum_v, 99.9):.3f}  "
      f"p99.99={np.percentile(lum_v, 99.99):.3f}  max={lum_v.max():.1f}")

# Cap at p99.9 — measure clipped vs original
cap = float(np.percentile(lum_v, 99.9))
v_cap = np.clip(v, None, cap)
print(f"\n=== with vibrt clipped to p99.9 = {cap:.3f} ===")
print(f"  v_cap mean = ({v_cap[..., 0].mean():.3f}, {v_cap[..., 1].mean():.3f}, {v_cap[..., 2].mean():.3f})")
print(f"  cycles mean= ({c[..., 0].mean():.3f}, {c[..., 1].mean():.3f}, {c[..., 2].mean():.3f})")
print(f"  ratio       = {v_cap.sum() / c.sum():.3f}")

# How many pixels are firefly (lum > 50, e.g.)?
n_firefly = int((lum_v > 50).sum())
firefly_energy = float(v[lum_v > 50].sum())
print(f"\n  vibrt pixels with lum > 50: n={n_firefly}, energy={firefly_energy:.0f} ({firefly_energy / v.sum() * 100:.1f}% of total)")

# Top 30 vibrt brightest pixels
print("\n=== top 30 vibrt-brightest pixels ===")
flat_idx = np.argsort(lum_v.ravel())[::-1][:30]
for i, idx in enumerate(flat_idx):
    y, x = divmod(int(idx), W)
    print(f"  #{i:>2}: y={y:>3} x={x:>3}   "
          f"vibrt={tuple(round(float(v[y, x, k]), 2) for k in range(3))}  "
          f"cycles={tuple(round(float(c[y, x, k]), 2) for k in range(3))}  "
          f"v/c={lum_v[y, x] / max(lum_c[y, x], 1e-6):.2f}")

# Also: bucket by cycles luminance to understand broad over
buckets = [(0, 0.001, "very_dark"), (0.001, 0.01, "dark"), (0.01, 0.05, "mid_low"),
           (0.05, 0.2, "mid"), (0.2, 1.0, "bright"), (1.0, 1e9, "super")]
print(f"\n=== bucket by cycles luminance ===")
print(f"{'bucket':>10} {'n':>8} {'sum_v':>10} {'sum_c':>10} {'ratio v/c':>10}")
for lo, hi, name in buckets:
    mask = (lum_c >= lo) & (lum_c < hi)
    if not mask.any():
        continue
    n = int(mask.sum())
    sv = float(v[mask].sum())
    sc = float(c[mask].sum())
    r = sv / max(sc, 1e-6)
    print(f"{name:>10} {n:>8d} {sv:>10.1f} {sc:>10.1f} {r:>10.3f}")
