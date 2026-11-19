"""Dump specific EXR pixel values for the floor-only test scene."""
import bpy
import numpy as np

for path, label in [
    ("c:/Users/hato2/Desktop/vibrt/test_scenes/lone_monk/_floor_vibrt.exr", "vibrt"),
    ("c:/Users/hato2/Desktop/vibrt/test_scenes/lone_monk/_floor_cycles.exr", "cycles"),
]:
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Linear Rec.709"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    arr = flat.reshape((h, w, 4))[..., :3]
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    yi, xi = np.unravel_index(lum.argmax(), lum.shape)
    print(f"\n{label}: shape={arr.shape}, peak={lum.max():.1f} @ pixel ({yi},{xi}) "
          f"RGB=({arr[yi,xi,0]:.1f}, {arr[yi,xi,1]:.1f}, {arr[yi,xi,2]:.1f})")
    samples = [
        ("top-edge center (sky high)", 0, w // 2),
        ("upper third (sky/horizon)", h // 4, w // 2),
        ("middle (horizon)", h // 2, w // 2),
        ("lower third (floor near)", 3 * h // 4, w // 2),
        ("bottom-edge center (floor close)", h - 1, w // 2),
        ("top-left", 0, 0),
        ("top-right", 0, w - 1),
        ("bottom-left", h - 1, 0),
        ("bottom-right", h - 1, w - 1),
    ]
    for label2, y, x in samples:
        print(f"  {label2:38s} ({y:3d},{x:3d}) "
              f"RGB=({arr[y,x,0]:8.2f}, {arr[y,x,1]:8.2f}, {arr[y,x,2]:8.2f})")
    bpy.data.images.remove(img)
