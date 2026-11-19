"""Locate the brightest pixel in the no-rotation Cycles HDRI bake and
compare against where the JPG-sun-uv (0.537, 0.804) should map per
forward (R=identity, so just uv = d.xy)."""
import bpy, numpy as np, math, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(path):
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = "Non-Color"
    arr = np.empty(img.size[0]*img.size[1]*4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    rgba = arr.reshape((img.size[1], img.size[0], 4))[::-1, :, :3]
    return np.where(rgba <= 0.04045, rgba/12.92, ((rgba+0.055)/1.055)**2.4)

eq = load(os.path.join(HERE, "test_scenes", "pabellon",
                        "_cycles_hdri_only_eq_norot.png"))
H, W, _ = eq.shape

luma = eq.mean(axis=2)
# Top 10 brightest pixels
flat = luma.flatten()
top = np.argsort(flat)[-20:][::-1]
print(f"\nTop 20 bright pixels in cycles no-rot bake:")
for k, idx in enumerate(top):
    yy, xx = divmod(idx, W)
    u, v = xx / W, yy / H
    val = tuple(round(float(x), 3) for x in eq[yy, xx])
    # World direction at this pixel
    th = v * math.pi; ph = u * 2 * math.pi
    d = (round(math.sin(th)*math.cos(ph), 3),
         round(math.sin(th)*math.sin(ph), 3),
         round(math.cos(th), 3))
    print(f"  #{k+1}: pixel=({xx},{yy}) uv=({u:.3f},{v:.3f}) value={val} dir={d}")

# JPG sun is at uv (0.537, 0.804). With identity rotation, FLAT projection
# means d.xy ≈ (0.537, 0.804) AND z = ±sqrt(1-0.537²-0.804²) = ±0.257
print()
for sign in (+1, -1):
    dz = sign * math.sqrt(max(0.0, 1.0 - 0.537**2 - 0.804**2))
    d = (0.537, 0.804, dz)
    th = math.acos(d[2])
    ph = math.atan2(d[1], d[0])
    if ph < 0: ph += 2 * math.pi
    print(f"predicted sun (z={dz:+.3f}): theta={math.degrees(th):.1f}°, "
          f"phi={math.degrees(ph):.1f}° → uv=({ph/(2*math.pi):.3f}, {th/math.pi:.3f})")
