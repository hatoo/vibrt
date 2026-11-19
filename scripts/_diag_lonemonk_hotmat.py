"""Identify which lone_monk materials still over-deliver (post brick fix).
Loads the current vibrt/cycles EXRs, finds worst-over bright pixels, ray_casts
to materials, aggregates over-energy by material."""
import sys
from pathlib import Path
import numpy as np
import bpy

HERE = Path(__file__).resolve().parent.parent
VE = HERE / "test_scenes" / "_eng_diff" / "lone_monk_vibrt.exr"
CE = HERE / "test_scenes" / "_eng_diff" / "lone_monk_cycles.exr"


def load(p):
    img = bpy.data.images.load(str(p))
    img.colorspace_settings.name = "Linear Rec.709"
    w, h = int(img.size[0]), int(img.size[1])
    fl = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(fl)
    bpy.data.images.remove(img)
    return fl.reshape(h, w, 4)[..., :3][::-1].copy(), w, h


v, w, h = load(VE)
c, _, _ = load(CE)
vlum = v @ np.array([0.2126, 0.7152, 0.0722], np.float32)
clum = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
over = vlum - clum
mask = (clum > 0.1) & (over > 0.1)
ys, xs = np.where(mask)
order = np.argsort(-over[ys, xs]); ys, xs = ys[order], xs[order]
print(f"over-delivering bright pixels: {len(ys)}  (analyzing top 500)")

scene = bpy.context.scene
dg = bpy.context.evaluated_depsgraph_get()
cam = scene.camera; cm = cam.matrix_world
tr, br, bl, tl = [cm @ p for p in cam.data.view_frame(scene=scene)]
origin = cm.translation
def rd(px, py):
    top = tl.lerp(tr, px); bot = bl.lerp(br, px)
    return (bot.lerp(top, 1 - py) - origin).normalized()

from collections import defaultdict
mo = defaultdict(float); mn = defaultdict(int)
N = min(500, len(ys))
for i in range(N):
    yy, xx = int(ys[i]), int(xs[i])
    d = rd((xx + 0.5) / w, (yy + 0.5) / h)
    hit, loc, nrm, idx, obj, mtx = scene.ray_cast(dg, origin, d)
    nm = "<miss/world>"
    if hit and obj and obj.material_slots and idx < len(obj.data.polygons):
        mi = obj.data.polygons[idx].material_index
        slot = obj.material_slots[mi] if mi < len(obj.material_slots) else None
        if slot and slot.material:
            nm = slot.material.name
    mo[nm] += float(over[yy, xx]); mn[nm] += 1

print("material                          over_sum   n")
for nm, ov in sorted(mo.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {nm:32s} {ov:8.3f} {mn[nm]:3d}")
print("=== DONE ===", flush=True)
