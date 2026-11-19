"""For Cycles' bright sun pixel (u=0.872, v=0.422 in the aligned
equirect), figure out the JPG pixel coords under each candidate UV
transformation, fetch the JPG value (in both bottom-up and top-down
storage), and compare against the Cycles equirect value (1.0, 1.0, 0.8)."""
import bpy, numpy as np, math, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(path, flip=True):
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = "Non-Color"
    arr = np.empty(img.size[0]*img.size[1]*4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    rgba = arr.reshape((img.size[1], img.size[0], 4))
    if flip:
        rgba = rgba[::-1]
    rgb = rgba[..., :3]
    rgb = np.where(rgb <= 0.04045, rgb/12.92, ((rgb+0.055)/1.055)**2.4)
    return rgb

jpg_path = bpy.path.abspath(bpy.data.images["DSC_8129.JPG"].filepath)
jpg_top = load(jpg_path, flip=True)
jpg_bot = load(jpg_path, flip=False)
jh, jw = jpg_top.shape[:2]

# Cycles bright pixel direction
u, v = 0.872, 0.422
th = v * math.pi; ph = u * 2 * math.pi
d = np.array([math.sin(th)*math.cos(ph), math.sin(th)*math.sin(ph), math.cos(th)])
print(f"world direction at cycles bright pixel: {tuple(round(float(x),3) for x in d)}")

rx, ry, rz = -1.414036, 0.685206, 0.228047
cx, sx = math.cos(rx), math.sin(rx); cy, sy = math.cos(ry), math.sin(ry); cz, sz = math.cos(rz), math.sin(rz)
R = np.array([
    [cy*cz,        sx*sy*cz - cx*sz,  cx*sy*cz + sx*sz],
    [cy*sz,        sx*sy*sz + cx*cz,  cx*sy*sz - sx*cz],
    [-sy,          sx*cy,             cx*cy],
])

candidates = {
    "(R · d)": R @ d,
    "(R · -d)": -(R @ d),
    "(R^T · d)": R.T @ d,
    "(R^T · -d)": -(R.T @ d),
}

print(f"\n{'transform':15s} {'(u,v)':>22s} {'top-down jpg value':>27s} {'bottom-up jpg value':>28s}")
for name, rd in candidates.items():
    uu = rd[0] % 1.0; vv = rd[1] % 1.0
    px = int(uu * jw); py = int(vv * jh)
    px = min(max(px, 0), jw-1); py = min(max(py, 0), jh-1)
    top_val = tuple(round(float(x), 3) for x in jpg_top[py, px])
    bot_val = tuple(round(float(x), 3) for x in jpg_bot[py, px])
    print(f"{name:15s} {f'({uu:.3f}, {vv:.3f})':>22s} {str(top_val):>27s} {str(bot_val):>28s}")

print(f"\nCycles equirect value at u=0.872, v=0.422: (1.0, 1.0, 0.799)  ← target")
