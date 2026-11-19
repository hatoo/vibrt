"""Sample classroom vibrt vs cycles by region to find where the gap
came from after the pure_diffuse change. Pure-Lambert vs Lambert+Spec
should mostly affect rounded geometry under grazing incidence — the
chairs, columns, body-of-radiator etc. Compare specific bands.
"""
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
v = np.asarray(Image.open(os.path.join(HERE, "test_scenes/classroom/preview.png")).convert("RGB"), dtype=np.float32) / 255.0
c = np.asarray(Image.open(os.path.join(HERE, "test_scenes/classroom/preview_cycles.png")).convert("RGB"), dtype=np.float32) / 255.0
print(f"shapes: vibrt={v.shape}, cycles={c.shape}")

def stats(label, region):
    y0, y1, x0, x1 = region
    cv = v[y0:y1, x0:x1, :].mean(axis=(0, 1))
    cc = c[y0:y1, x0:x1, :].mean(axis=(0, 1))
    diff = cv - cc
    print(f"  {label:25s}: vibrt=({cv[0]:.3f}, {cv[1]:.3f}, {cv[2]:.3f}) "
          f"cycles=({cc[0]:.3f}, {cc[1]:.3f}, {cc[2]:.3f}) "
          f"diff=({diff[0]:+.3f}, {diff[1]:+.3f}, {diff[2]:+.3f})")

H, W, _ = v.shape
# Top band (likely ceiling, lights), bottom band (floor), left/right (walls).
print("\n=== regions ===")
stats("top 0-15% (ceiling)",      (0, int(H * 0.15), 0, W))
stats("middle 30-50% (mid wall)", (int(H * 0.30), int(H * 0.50), 0, W))
stats("bottom 70-100% (floor)",   (int(H * 0.70), H, 0, W))
stats("left 0-25% (window-side)", (0, H, 0, int(W * 0.25)))
stats("right 75-100% (door-side)", (0, H, int(W * 0.75), W))
stats("center 30-70%",             (int(H * 0.30), int(H * 0.70), int(W * 0.30), int(W * 0.70)))
