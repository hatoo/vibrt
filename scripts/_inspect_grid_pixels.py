"""Inspect specific equirect pixels in both vibrt's mixed grid and
Cycles' equirect bake. Compute the world directions that should land
on the JPG sun (uv=0.537, 0.804) per `R·d`, and compare brightness."""
import bpy, numpy as np, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(path):
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = "Non-Color"
    arr = np.empty(img.size[0]*img.size[1]*4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    rgba = arr.reshape((img.size[1], img.size[0], 4))[::-1, :, :3]
    rgb = np.where(rgba <= 0.04045, rgba/12.92, ((rgba+0.055)/1.055)**2.4)
    return rgb

vg = load(os.path.join(HERE, "test_scenes", "pabellon", "_mixed_grid.png"))
cg = load(os.path.join(HERE, "test_scenes", "pabellon", "_cycles_equirect_world.png"))
print(f"vibrt grid {vg.shape}  cycles {cg.shape}")

H, W, _ = vg.shape

# Predicted sun world directions (computed from R^T applied to JPG sun uv)
candidates = [
    ("sun_+ (u=0.85, v=0.23)", (0.85, 0.23)),
    ("sun_- (u=0.97, v=0.26)", (0.97, 0.26)),
    ("brightest in cycles  ", None),
]
# Find brightest pixel in cycles
luma = cg.mean(axis=2)
yy, xx = np.unravel_index(np.argmax(luma), luma.shape)
candidates[2] = (candidates[2][0], (xx/W, yy/H))
print(f"\ncycles brightest at u={xx/W:.3f}, v={yy/H:.3f}, value={tuple(round(float(x),3) for x in cg[yy,xx])}")

print(f"\n{'pixel':30s} {'vibrt rgb':>30s} {'cycles rgb':>30s}")
for name, (u, v) in candidates:
    px = int(u * W); py = int(v * H)
    px = min(px, W-1); py = min(py, H-1)
    vr = tuple(round(float(x), 3) for x in vg[py, px])
    cr = tuple(round(float(x), 3) for x in cg[py, px])
    print(f"{name:30s} {str(vr):>30s} {str(cr):>30s}")
