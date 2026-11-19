"""Locate the sun pixel in DSC_8129.JPG and figure out which (uv) it
sits at. Then for the world direction where Cycles' aligned equirect
shows the sun (around u=0.85, v=0.3 → world dir (+0.48, -0.65, +0.59)),
compute (R·d).xy and (R·-d).xy and see which lands on the JPG sun."""
import bpy, numpy as np

img = bpy.data.images.get("DSC_8129.JPG")
if img is None:
    raise SystemExit("no DSC_8129.JPG")

# Pull pixels in linear (Non-Color) space.
prev = img.colorspace_settings.name
img.colorspace_settings.name = "Non-Color"
try:
    arr = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
    img.pixels.foreach_get(arr)
finally:
    img.colorspace_settings.name = prev

w, h = img.size
rgba = arr.reshape((h, w, 4))[::-1]  # foreach_get is bottom-up
rgb = rgba[..., :3]
# Convert sRGB → linear so brightness is comparable to whatever Cycles sees.
# Actually JPG is stored in sRGB; Non-Color disables Cycles' EOTF, so we're
# reading raw 8-bit values. Linearise manually.
rgb_lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
print(f"image {w}x{h}")
print(f"raw mean rgb = {tuple(round(float(x),3) for x in rgb.mean(axis=(0,1)))}")
print(f"linear mean rgb = {tuple(round(float(x),3) for x in rgb_lin.mean(axis=(0,1)))}")

# Find the brightest pixel (sun).
lum = rgb_lin.mean(axis=2)
yy, xx = np.unravel_index(np.argmax(lum), lum.shape)
print(f"brightest pixel at (x, y) = ({xx}, {yy}) → uv = ({xx/w:.4f}, {yy/h:.4f})")
print(f"   value = {tuple(round(float(rgb_lin[yy, xx, c]), 3) for c in range(3))}")

# Top 20 brightest cluster (in sky)
threshold = np.percentile(lum, 99.99)
mask = lum >= threshold
ys, xs = np.where(mask)
print(f"top 0.01% mean position uv = ({xs.mean()/w:.4f}, {ys.mean()/h:.4f}), count = {len(xs)}")

# Now compute target sun direction in world (from earlier reasoning):
# u=0.85, v=0.3 in vibrt's grid → theta=0.3π, phi=0.85*2π
import math
def dir_from_uv(u, v):
    th = v * math.pi
    ph = u * 2 * math.pi
    return np.array([math.sin(th)*math.cos(ph),
                     math.sin(th)*math.sin(ph),
                     math.cos(th)])

# Mapping rotation
rx, ry, rz = -1.4140, 0.6852, 0.2280
def euler_xyz(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([
        [cy*cz,        sx*sy*cz - cx*sz,  cx*sy*cz + sx*sz],
        [cy*sz,        sx*sy*sz + cx*cz,  cx*sy*sz - sx*cz],
        [-sy,          sx*cy,             cx*cy],
    ])

R = euler_xyz(rx, ry, rz)
print(f"Mapping rotation R =\n{R}")

# Find all pixels in the Cycles equirect that are very bright; for each,
# compute the corresponding world direction, then (R*d).xy and (R*-d).xy.
def show_at(u, v):
    d = dir_from_uv(u, v)
    Rd = R @ d
    nRd = -Rd  # = R @ -d
    print(f"  vibrt grid pixel (u={u:.3f}, v={v:.3f}) → world dir {tuple(round(float(x),3) for x in d)}")
    print(f"     R·d  = ({Rd[0]:+.3f}, {Rd[1]:+.3f}) → repeat-uv = ({Rd[0]%1:.3f}, {Rd[1]%1:.3f}) → JPG pixel ({int(Rd[0]%1*w)}, {int(Rd[1]%1*h)})")
    print(f"     R·-d = ({nRd[0]:+.3f}, {nRd[1]:+.3f}) → repeat-uv = ({nRd[0]%1:.3f}, {nRd[1]%1:.3f}) → JPG pixel ({int(nRd[0]%1*w)}, {int(nRd[1]%1*h)})")

print("\n# Sun in Cycles equirect appears around u=0.85, v=0.3:")
show_at(0.85, 0.30)
print("# Some other test directions:")
for (u, v) in [(0.5, 0.5), (0.0, 0.5), (0.5, 0.0)]:
    show_at(u, v)
