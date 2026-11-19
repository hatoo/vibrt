"""Find the brightest pixel in the saved bake PNG and report its (theta,
phi) under the kernel's convention `theta = v*pi`, `phi = (u - 0)*2pi`."""
import os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import bpy
img = bpy.data.images.load(os.path.join(HERE, "test_scenes", "pabellon", "_skybake_4096x2048.png"))
img.colorspace_settings.name = "Non-Color"
w, h = int(img.size[0]), int(img.size[1])
flat = np.empty(w * h * 4, dtype=np.float32)
img.pixels.foreach_get(flat)
rgba = flat.reshape((h, w, 4))
lum = 0.2126 * rgba[..., 0] + 0.7152 * rgba[..., 1] + 0.0722 * rgba[..., 2]

# Top-K brightest cluster locations
flat_lum = lum.flatten()
order = np.argsort(flat_lum)[::-1]
print(f"image is {w}x{h}, max lum={flat_lum.max():.3f}")
print("top-5 bright pixels:")
import math
for k in range(5):
    idx = order[k]
    y = idx // w
    x = idx % w
    u = (x + 0.5) / w
    v = (y + 0.5) / h
    # Note: PNG is loaded top-up by bpy.data.images, but
    # `image.pixels.foreach_get` is bottom-up. So row 0 in the array is
    # the bottom of the image. Translate to top-down for human reading.
    y_top = h - 1 - y
    v_top = (y_top + 0.5) / h
    # Kernel convention:
    #   theta = v * pi (where v is in [0, 1] from top of equirect)
    #   phi   = u * 2pi (in [0, 2pi))
    # But the saved PNG had bottom-up pixels — for a "top-down" reader
    # who looks at the rendered file, top y=0 is zenith (kernel theta=0).
    print(f"  rank {k}: x={x} y(bottom-up)={y} y_top={y_top}  "
          f"u={u:.4f} v_bu={v:.4f} v_top={v_top:.4f}  lum={flat_lum[idx]:.3f}  "
          f"phi_bake={u*360-180:.1f}deg  theta_top={v_top*180:.1f}deg")
