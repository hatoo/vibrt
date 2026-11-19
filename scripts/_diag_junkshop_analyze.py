"""Analyze the existing junk_shop EXRs: is the 1.46x over-delivery
firefly-driven or broad? Report overall ratio, ratio after clipping hot
pixels, and per-brightness-bucket ratios. Fast, no re-render.
"""
import sys
from pathlib import Path
import numpy as np
import bpy

HERE = Path(__file__).resolve().parent.parent
VE = HERE / "test_scenes" / "_eng_diff" / "junk_shop_vibrt.exr"
CE = HERE / "test_scenes" / "_eng_diff" / "junk_shop_cycles.exr"


def load(p):
    img = bpy.data.images.load(str(p))
    img.colorspace_settings.name = "Linear Rec.709"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    bpy.data.images.remove(img)
    return flat.reshape(h, w, 4)[..., :3][::-1].copy()


v = load(VE); c = load(CE)
h, w, _ = v.shape
print(f"junk_shop {w}x{h}")
vf = v.reshape(-1, 3); cf = c.reshape(-1, 3)
print(f"overall  vibrt={np.round(vf.mean(0),4)} cycles={np.round(cf.mean(0),4)} ratio={np.round(vf.mean(0)/np.maximum(cf.mean(0),1e-9),3)}")

vlum = vf @ np.array([0.2126, 0.7152, 0.0722], np.float32)
clum = cf @ np.array([0.2126, 0.7152, 0.0722], np.float32)

# Clip top-N% brightest vibrt pixels and recompute ratio.
for pct in (0.0, 0.1, 0.5, 1.0, 2.0):
    thr = np.percentile(vlum, 100 - pct) if pct > 0 else np.inf
    m = vlum <= thr
    r = vf[m].mean(0) / np.maximum(cf[m].mean(0), 1e-9)
    print(f"  clip top {pct:.1f}% (vlum<= {thr:.2f}): ratio={np.round(r,3)}  (n={m.sum()})")

# Per-cycles-luminance bucket ratio (where does the over concentrate?).
print("\nbucket (cycles lum)   n       vibrt_mean   cycles_mean   ratio   over_energy_share")
edges = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1e9]
tot_over = 0.0
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    mask = (clum >= lo) & (clum < hi)
    n = int(mask.sum())
    if n == 0:
        continue
    vm = vf[mask].mean(0); cm = cf[mask].mean(0)
    over = (vf[mask].sum(0) - cf[mask].sum(0)).sum()
    tot_over += max(over, 0)
    rows.append((lo, hi, n, vm, cm, over))
for lo, hi, n, vm, cm, over in rows:
    share = over / max(tot_over, 1e-9)
    tag = f"[{lo:.2f},{hi:.2f})"
    print(f"{tag:20s} {n:7d}  {np.round(vm,3)} {np.round(cm,3)} {np.round(vm.mean()/max(cm.mean(),1e-9),2):>6}  {share:+.2%}")
print("=== DONE ===", flush=True)
