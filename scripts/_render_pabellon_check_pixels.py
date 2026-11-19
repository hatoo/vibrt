"""Read the pabellon preview and dump the pixel values across a few
columns of the foreground water area. Compare to the cycles preview to
quantify the underwater visibility gap."""
import bpy, os, numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(path):
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "sRGB"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    return flat.reshape((h, w, 4)), w, h

vibrt, vw, vh = load(os.path.join(HERE, "test_scenes", "pabellon", "preview.png"))
cyc, cw, ch = load(os.path.join(HERE, "test_scenes", "pabellon", "preview_cycles.png"))
print(f"vibrt: {vw}x{vh}, cycles: {cw}x{ch}")

# Image is bottom-up. We want to look at FOREGROUND water — upper rows of the
# array (= bottom of image). Row y from buffer-bottom = (vh - y_from_top - 1).
# Cycles preview is the same dim... let me sample at y_from_top = 200, 220, 240
# (in the lower third of the image where the water is).

# Compute mean brightness in foreground water (broader sampling).
import math
y_top_lo, y_top_hi = 350, 600  # foreground pool area
x_lo, x_hi = 100, 1200
fg_v = vibrt[vh-1-y_top_hi:vh-1-y_top_lo+1, x_lo:x_hi, :3]
fg_c = cyc[vh-1-y_top_hi:vh-1-y_top_lo+1, x_lo:x_hi, :3]
print(f"foreground water box: {fg_v.shape[0]}x{fg_v.shape[1]}")
print(f"  vibrt mean rgb = {tuple(round(float(x),3) for x in fg_v.mean(axis=(0,1)))}")
print(f"  cycles mean rgb = {tuple(round(float(x),3) for x in fg_c.mean(axis=(0,1)))}")
print(f"  vibrt p50 = {tuple(round(float(np.percentile(fg_v[..., c], 50)),3) for c in range(3))}")
print(f"  cycles p50 = {tuple(round(float(np.percentile(fg_c[..., c], 50)),3) for c in range(3))}")
print(f"  vibrt p10/p90 = {tuple(round(float(np.percentile(fg_v[..., c], 10)),3) for c in range(3))} / {tuple(round(float(np.percentile(fg_v[..., c], 90)),3) for c in range(3))}")
print(f"  cycles p10/p90 = {tuple(round(float(np.percentile(fg_c[..., c], 10)),3) for c in range(3))} / {tuple(round(float(np.percentile(fg_c[..., c], 90)),3) for c in range(3))}")
# Standard deviation = visual contrast / pebble visibility
print(f"  vibrt std = {tuple(round(float(fg_v[..., c].std()),3) for c in range(3))}")
print(f"  cycles std = {tuple(round(float(fg_c[..., c].std()),3) for c in range(3))}")
