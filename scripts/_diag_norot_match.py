"""For Cycles' no-rotation bright bake pixel (uv 0.654, 0.283 → world
dir (-0.44, -0.64, 0.63)), check what the JPG holds at uv (-0.44, -0.64)
mod 1 = (0.56, 0.36) under both top-down and bottom-up stores. Then
verify with mirror points (d.xy negated, etc.)."""
import bpy, numpy as np, math, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(path, flip=True, linearize=True):
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = "Non-Color"
    arr = np.empty(img.size[0]*img.size[1]*4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    rgba = arr.reshape((img.size[1], img.size[0], 4))
    if flip:
        rgba = rgba[::-1]
    rgb = rgba[..., :3]
    if linearize:
        rgb = np.where(rgb <= 0.04045, rgb/12.92, ((rgb+0.055)/1.055)**2.4)
    return rgb

jpg_path = bpy.path.abspath(bpy.data.images["DSC_8129.JPG"].filepath)
jpg_top = load(jpg_path, flip=True)
jpg_bot = load(jpg_path, flip=False)
jh, jw = jpg_top.shape[:2]

# Cycles bright pixel #1: uv=(0.654, 0.283) value=(0.896, 0.88, 0.839)
target_value = (0.896, 0.88, 0.839)
d = (-0.44, -0.641, 0.63)

print(f"Target: cycles bright pixel value = {target_value}")
print(f"World direction: {d}")
print(f"\n# With R = identity, (R·d).xy = d.xy = (-0.44, -0.641)")

candidates = {
    "(d.x, d.y)":          ( d[0],  d[1]),
    "(-d.x, d.y)":         (-d[0],  d[1]),
    "(d.x, -d.y)":         ( d[0], -d[1]),
    "(-d.x, -d.y)":        (-d[0], -d[1]),
    "(d.y, d.x)":          ( d[1],  d[0]),
    "(d.y, -d.x)":         ( d[1], -d[0]),
    "(-d.y, d.x)":         (-d[1],  d[0]),
}
print(f"\n{'transform':18s} {'(u,v)':>22s} {'top-down':>27s} {'bottom-up':>27s}")
for name, (u, v) in candidates.items():
    uu = u % 1.0; vv = v % 1.0
    px = int(uu * jw); py = int(vv * jh)
    px = min(max(px, 0), jw-1); py = min(max(py, 0), jh-1)
    top_val = tuple(round(float(x), 3) for x in jpg_top[py, px])
    bot_val = tuple(round(float(x), 3) for x in jpg_bot[py, px])
    print(f"{name:18s} {f'({uu:.3f}, {vv:.3f})':>22s} {str(top_val):>27s} {str(bot_val):>27s}")
