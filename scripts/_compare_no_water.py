"""Compare _no_water.png vs cycles to see pebble brightness without water."""
import bpy, os, numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(path):
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "sRGB"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    return flat.reshape((h, w, 4))[..., :3]

vibrt_nw = load(os.path.join(HERE, "test_scenes", "pabellon", "_no_water.png"))
cyc = load(os.path.join(HERE, "test_scenes", "pabellon", "preview_cycles.png"))

# Same foreground pool box
y_top_lo, y_top_hi = 350, 600
x_lo, x_hi = 100, 1200
h_v = vibrt_nw.shape[0]
fg_v = vibrt_nw[h_v-1-y_top_hi:h_v-1-y_top_lo+1, x_lo:x_hi, :3]
fg_c = cyc[h_v-1-y_top_hi:h_v-1-y_top_lo+1, x_lo:x_hi, :3]
print(f"foreground box: {fg_v.shape}")
print(f"  vibrt no-water mean = {tuple(round(float(c),3) for c in fg_v.mean(axis=(0,1)))}")
print(f"  cycles WITH water  mean = {tuple(round(float(c),3) for c in fg_c.mean(axis=(0,1)))}")
print(f"  vibrt no-water p50 = {tuple(round(float(np.percentile(fg_v[..., c], 50)),3) for c in range(3))}")
print(f"  vibrt no-water p10 = {tuple(round(float(np.percentile(fg_v[..., c], 10)),3) for c in range(3))}")
