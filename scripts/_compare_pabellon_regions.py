"""Region-level diff for pabellon vs Cycles to spot where the +0.058
B-channel over-blueness comes from after the mixed-envmap path.
"""
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
v = np.asarray(Image.open(os.path.join(HERE, "test_scenes/pabellon/preview.png")).convert("RGB"), dtype=np.float32) / 255.0
c = np.asarray(Image.open(os.path.join(HERE, "test_scenes/pabellon/preview_cycles.png")).convert("RGB"), dtype=np.float32) / 255.0
print(f"shapes: vibrt={v.shape}, cycles={c.shape}")
if v.shape != c.shape:
    from PIL import Image as PI
    c_im = PI.fromarray((c*255).astype(np.uint8)).resize((v.shape[1], v.shape[0]), PI.LANCZOS)
    c = np.asarray(c_im, dtype=np.float32)/255.0

def stats(label, region):
    y0, y1, x0, x1 = region
    cv = v[y0:y1, x0:x1, :].mean(axis=(0,1))
    cc = c[y0:y1, x0:x1, :].mean(axis=(0,1))
    d = cv - cc
    print(f"  {label:25s}: vibrt=({cv[0]:.3f},{cv[1]:.3f},{cv[2]:.3f}) "
          f"cycles=({cc[0]:.3f},{cc[1]:.3f},{cc[2]:.3f}) "
          f"diff=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})")

H, W, _ = v.shape
print("\n=== regions ===")
stats("top 0-15% (sky)",            (0, int(H*0.15), 0, W))
stats("top-mid 15-35%",              (int(H*0.15), int(H*0.35), 0, W))
stats("mid 35-55% (building)",       (int(H*0.35), int(H*0.55), 0, W))
stats("low-mid 55-75%",              (int(H*0.55), int(H*0.75), 0, W))
stats("bottom 75-100% (water)",      (int(H*0.75), H, 0, W))
stats("center 30-70%",               (0, H, int(W*0.30), int(W*0.70)))
