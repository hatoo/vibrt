"""For each bright pixel in the Cycles aligned equirect, the
corresponding JPG pixel must (approximately) hold the same colour. Use
this to back-out which transformation Cycles applied to the
world-direction before sampling the JPG."""
import bpy, numpy as np, math, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_image(path, lin=True):
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = "Non-Color"
    arr = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    rgb = arr.reshape((img.size[1], img.size[0], 4))[::-1, :, :3]
    if lin:
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return rgb

eq = load_image(os.path.join(HERE, "test_scenes", "pabellon",
                              "_cycles_equirect_world.png"))
# Reload JPG via path so it's freshly loaded into bpy.data.
jpg_path = bpy.path.abspath(bpy.data.images["DSC_8129.JPG"].filepath)
print(f"jpg from {jpg_path}")
jpg = load_image(jpg_path)
print(f"eq {eq.shape} jpg {jpg.shape}")

# Mapping rotation
rx, ry, rz = -1.4140, 0.6852, 0.2280
cx, sx = math.cos(rx), math.sin(rx); cy, sy = math.cos(ry), math.sin(ry); cz, sz = math.cos(rz), math.sin(rz)
R = np.array([
    [cy*cz,        sx*sy*cz - cx*sz,  cx*sy*cz + sx*sz],
    [cy*sz,        sx*sy*sz + cx*cz,  cx*sy*sz - sx*cz],
    [-sy,          sx*cy,             cx*cy],
])

eh, ew = eq.shape[:2]
jh, jw = jpg.shape[:2]

# For each equirect pixel, compute world direction, four candidate UVs,
# fetch JPG colour, compare to eq colour. Find the candidate with the
# smallest mean error.
us = (np.arange(ew) + 0.5) / ew
vs = (np.arange(eh) + 0.5) / eh
uu, vv = np.meshgrid(us, vs)
phi = uu * 2 * math.pi
theta = vv * math.pi
dirs = np.stack([
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta),
], axis=-1).reshape(-1, 3)

def fetch(uv_x, uv_y):
    """Bilinear-ish fetch into jpg with REPEAT extension."""
    fx = (uv_x % 1.0) * jw
    fy = (uv_y % 1.0) * jh
    xi = np.clip(fx.astype(int), 0, jw - 1)
    yi = np.clip(fy.astype(int), 0, jh - 1)
    return jpg[yi, xi, :]

eq_flat = eq.reshape(-1, 3)

candidates = {
    "(R · d)": (R @ dirs.T).T,
    "(R · -d)": -(R @ dirs.T).T,
    "(R^T · d)": (R.T @ dirs.T).T,
    "(R^T · -d)": -(R.T @ dirs.T).T,
}

# Mask out areas where Cycles is very dark (sky-dominated, less reliable).
brightness = eq_flat.mean(axis=1)
mask = brightness > 0.05

for name, rotated in candidates.items():
    sampled = fetch(rotated[:, 0], rotated[:, 1])
    err = np.abs(sampled - eq_flat).mean(axis=1)
    print(f"{name}: mean err (mask) = {err[mask].mean():.4f},  "
          f"corr = {np.corrcoef(sampled[mask].mean(axis=1), eq_flat[mask].mean(axis=1))[0,1]:.4f}")

# Try transposed-then-swap-xy and other swaps
extra = {
    "(R · d) swap xy": (R @ dirs.T).T[:, [1, 0, 2]],
    "(R · -d) swap xy": -(R @ dirs.T).T[:, [1, 0, 2]],
    "(R^T · d) swap xy": (R.T @ dirs.T).T[:, [1, 0, 2]],
    "(R^T · -d) swap xy": -(R.T @ dirs.T).T[:, [1, 0, 2]],
}
print()
for name, rotated in extra.items():
    sampled = fetch(rotated[:, 0], rotated[:, 1])
    err = np.abs(sampled - eq_flat).mean(axis=1)
    print(f"{name}: mean err (mask) = {err[mask].mean():.4f},  "
          f"corr = {np.corrcoef(sampled[mask].mean(axis=1), eq_flat[mask].mean(axis=1))[0,1]:.4f}")
