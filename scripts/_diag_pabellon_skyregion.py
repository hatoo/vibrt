"""Decompose pabellon's B-over: compare the direct-world (sky) region vs the
rest between the existing vibrt/cycles EXRs. If the sky region itself is
B-over, it's a sky-bake colour issue; if only lit surfaces are, it's indirect.
Run with plain python (imports OpenEXR via imageio, or falls back to Blender).
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
VE = HERE / "test_scenes" / "_eng_diff" / "pabellon_vibrt.exr"
CE = HERE / "test_scenes" / "_eng_diff" / "pabellon_cycles.exr"


def load_exr_blender(p):
    import bpy
    img = bpy.data.images.load(str(p))
    img.colorspace_settings.name = "Linear Rec.709"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    bpy.data.images.remove(img)
    # Blender loads bottom-up; flip to top-down.
    return flat.reshape(h, w, 4)[..., :3][::-1].copy()


v = load_exr_blender(VE)
c = load_exr_blender(CE)
h, w, _ = v.shape
print(f"image {w}x{h}")


def stats(name, vv, cc):
    vm = vv.reshape(-1, 3).mean(axis=0)
    cm = cc.reshape(-1, 3).mean(axis=0)
    ratio = vm / np.maximum(cm, 1e-9)
    print(f"  {name:12s} vibrt={np.round(vm,4)} cycles={np.round(cm,4)} ratio={np.round(ratio,3)}")


# Top 15% rows = sky region (pabellon camera looks out; sky at top).
top = int(h * 0.15)
print("=== region decomposition ===")
stats("sky(top15%)", v[:top], c[:top])
stats("mid(15-60%)", v[top:int(h*0.6)], c[top:int(h*0.6)])
stats("lower(60%+)", v[int(h*0.6):], c[int(h*0.6):])
stats("full", v, c)

# Identify very-bright (direct-world) pixels by cycles luminance and compare.
clum = c @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
for lo, hi, tag in [(0.0, 0.05, "dark"), (0.05, 0.2, "mid"), (0.2, 1.0, "bright"), (1.0, 1e9, "hot")]:
    mask = (clum >= lo) & (clum < hi)
    n = int(mask.sum())
    if n == 0:
        continue
    stats(f"{tag}({n})", v[mask], c[mask])

print("=== DONE ===", flush=True)
