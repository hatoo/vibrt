"""Sample average colour over a fixed rectangle in vibrt vs cycles preview
PNGs and print R/G, B/G ratios.
"""
import sys, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(HERE, "test_scenes", "lone_monk")

def stats(path, region):
    p = os.path.join(out_dir, path)
    if not os.path.exists(p):
        print(f"  (missing: {p})")
        return None
    img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = img.shape
    x0, y0, x1, y1 = region
    crop = img[y0:y1, x0:x1, :]
    mean = crop.mean(axis=(0, 1))
    return mean

regions = {
    # x0, y0, x1, y1 in pixel coords on the 600x450 preview.
    "directly-lit-mid-distance": (260, 180, 360, 220),  # central facade
    "shadowed-foreground": (100, 350, 200, 410),         # foreground bricks
    "archway-ceiling": (170, 30, 430, 70),               # arched ceiling
}

for region_name, region in regions.items():
    print(f"\n=== {region_name} {region} ===")
    for label, fname in [
        ("vibrt+composit", "preview.png"),
        ("cycles+composit", "preview_cycles.png"),
        ("vibrt-RAW", "preview_raw_vibrt.png"),
        ("cycles-RAW", "preview_raw_cycles.png"),
    ]:
        m = stats(fname, region)
        if m is None:
            continue
        rg = m[0] / max(m[1], 1e-9)
        bg = m[2] / max(m[1], 1e-9)
        print(f"  {label:18s}: rgb=({m[0]:.4f}, {m[1]:.4f}, {m[2]:.4f}) "
              f"R/G={rg:.4f} B/G={bg:.4f}")
