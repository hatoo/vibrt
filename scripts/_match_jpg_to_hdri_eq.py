"""Compare each candidate UV transformation (R·d / R·-d / R^T·d /
R^T·-d) against the clean Cycles HDRI-only equirect bake. The
transformation with the lowest pixel error is the one Cycles actually
uses."""
import bpy, numpy as np, math, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_image(path, lin=True, flip=True):
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = "Non-Color"
    arr = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    rgb = arr.reshape((img.size[1], img.size[0], 4))[:, :, :3]
    if flip:
        rgb = rgb[::-1]
    if lin:
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return rgb

# Cycles equirect: rendered to a PNG file, displayed top-down. Flip
# foreach_get's bottom-up to top-down so eq[0] is the actual top row.
eq = load_image(os.path.join(HERE, "test_scenes", "pabellon",
                              "_cycles_hdri_only_eq.png"))
# JPG: same here — the rendered PNG comparison is in top-down.
jpg_path = bpy.path.abspath(bpy.data.images["DSC_8129.JPG"].filepath)
jpg = load_image(jpg_path)
# Also build a NO-FLIP variant (matches what scene_loader.rs has)
jpg_bottom_up = load_image(jpg_path, flip=False)
print(f"eq {eq.shape} jpg {jpg.shape}")

rx, ry, rz = -1.414036, 0.685206, 0.228047
cx, sx = math.cos(rx), math.sin(rx); cy, sy = math.cos(ry), math.sin(ry); cz, sz = math.cos(rz), math.sin(rz)
R = np.array([
    [cy*cz,        sx*sy*cz - cx*sz,  cx*sy*cz + sx*sz],
    [cy*sz,        sx*sy*sz + cx*cz,  cx*sy*sz - sx*cz],
    [-sy,          sx*cy,             cx*cy],
])

eh, ew = eq.shape[:2]
jh, jw = jpg.shape[:2]
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

def fetch_bilinear(uv_x, uv_y, src=None):
    src = src if src is not None else jpg
    fx = (uv_x % 1.0) * jw - 0.5
    fy = (uv_y % 1.0) * jh - 0.5
    x0 = np.floor(fx).astype(int); y0 = np.floor(fy).astype(int)
    dx = fx - x0; dy = fy - y0
    x0w = x0 % jw; x1w = (x0 + 1) % jw
    y0w = y0 % jh; y1w = (y0 + 1) % jh
    c00 = src[y0w, x0w]; c10 = src[y0w, x1w]
    c01 = src[y1w, x0w]; c11 = src[y1w, x1w]
    c0 = c00 * (1 - dx[:, None]) + c10 * dx[:, None]
    c1 = c01 * (1 - dx[:, None]) + c11 * dx[:, None]
    return c0 * (1 - dy[:, None]) + c1 * dy[:, None]

eq_flat = eq.reshape(-1, 3)

# Build sign / transpose / axis-swap variants.
def variants():
    base = R @ dirs.T
    yield "(R · d)", base.T
    yield "(R · -d)", -base.T
    yield "(R^T · d)", (R.T @ dirs.T).T
    yield "(R^T · -d)", -(R.T @ dirs.T).T
    # Try negating only x or only y of the result.
    yield "(R · d) negate x", base.T * np.array([-1, 1, 1])
    yield "(R · d) negate y", base.T * np.array([1, -1, 1])
    yield "(R · -d) negate y", -base.T * np.array([1, -1, 1])
    yield "(R^T · d) negate x", (R.T @ dirs.T).T * np.array([-1, 1, 1])
    yield "(R^T · d) negate y", (R.T @ dirs.T).T * np.array([1, -1, 1])

print(f"\n=== Using TOP-DOWN flipped JPG (matches PIL display) ===")
print(f"{'transform':30s} {'mean_err':>10s} {'corr':>10s} {'best?':>10s}")
best_err = 1e9
best_name = None
for name, rot in variants():
    sampled = fetch_bilinear(rot[:, 0], rot[:, 1])
    err = np.abs(sampled - eq_flat).mean()
    cor = np.corrcoef(sampled.mean(axis=1), eq_flat.mean(axis=1))[0, 1]
    flag = ""
    if err < best_err:
        best_err = err; best_name = name; flag = " <- best"
    print(f"{name:30s} {err:>10.4f} {cor:>10.4f}{flag}")

print(f"\n=== Using BOTTOM-UP raw JPG (matches scene_loader.rs raw foreach_get layout) ===")
best_err = 1e9
best_name = None
for name, rot in variants():
    sampled = fetch_bilinear(rot[:, 0], rot[:, 1], src=jpg_bottom_up)
    err = np.abs(sampled - eq_flat).mean()
    cor = np.corrcoef(sampled.mean(axis=1), eq_flat.mean(axis=1))[0, 1]
    flag = ""
    if err < best_err:
        best_err = err; best_name = name; flag = " <- best"
    print(f"{name:30s} {err:>10.4f} {cor:>10.4f}{flag}")

print(f"\nbest match: {best_name} (err={best_err:.4f})")
