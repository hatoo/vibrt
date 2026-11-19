"""Compare pixel values from a Non-Color tagged JPG vs an sRGB tagged
JPG to see what Blender returns via `pixels.foreach_get`. The hypothesis
is that `church brick`'s diff texture is mistakenly tagged Non-Color
(other bricks are sRGB), causing vibrt to read raw sRGB-encoded values
as if they were linear.
"""
import bpy
import numpy as np

samples = [
    ("castle_brick_02_red_diff_1k.jpg", "church brick (DIFF, sRGB-encoded JPG)"),
    ("medieval_wall_01_diff_1k.jpg", "brick clay (DIFF)"),
    ("brick_floor_diff_1k.jpg", "stone pavement (DIFF)"),
    ("PaintedPlaster014_1K_Color.jpg", "plaster med old (Color)"),
]

for img_name, label in samples:
    img = bpy.data.images.get(img_name)
    if img is None:
        print(f"-- {label}: not found")
        continue
    if img.size[0] == 0:
        img.update()
    w, h = int(img.size[0]), int(img.size[1])
    cs = img.colorspace_settings.name
    pixels = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(pixels)
    arr = pixels.reshape((h, w, 4))[..., :3]
    mean = arr.mean(axis=(0, 1))
    p50 = np.median(arr, axis=(0, 1))
    p95 = np.percentile(arr, 95, axis=(0, 1))
    print(f"-- {label}")
    print(f"   image: {img.filepath}")
    print(f"   colorspace = {cs!r}")
    print(f"   shape = {arr.shape}")
    print(f"   mean RGB = ({mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f})")
    print(f"   median   = ({p50[0]:.3f}, {p50[1]:.3f}, {p50[2]:.3f})")
    print(f"   p95      = ({p95[0]:.3f}, {p95[1]:.3f}, {p95[2]:.3f})")
