"""Measure linear-space pixel means for Cycles' aligned equirect bake
of pabellon's full mixed world, vs vibrt's host-rasterised grid. Tells
us whether the colour gap is in the rasterisation or downstream."""
import bpy, numpy as np, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_linear_png(path):
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = "Non-Color"
    arr = np.empty(img.size[0]*img.size[1]*4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    rgb = arr.reshape((img.size[1], img.size[0], 4))[::-1, :, :3]
    return rgb

cycles_full = load_linear_png(os.path.join(HERE, "test_scenes", "pabellon",
                                            "_cycles_equirect_world.png"))
cycles_hdri = load_linear_png(os.path.join(HERE, "test_scenes", "pabellon",
                                            "_cycles_hdri_only_eq.png"))
vibrt_grid = load_linear_png(os.path.join(HERE, "test_scenes", "pabellon",
                                           "_mixed_grid.png"))

# These are PNG-encoded pixels (not linearised — view_transform=Standard
# means pixel values ARE the linear radiance, but PNG storage uses sRGB
# encoding by default, so we should sRGB→linear them).
def to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb/12.92, ((rgb+0.055)/1.055)**2.4)

cf = to_linear(cycles_full)
ch = to_linear(cycles_hdri)
vg = to_linear(vibrt_grid)

print(f"Cycles full eq mean (linear): {tuple(round(float(x),4) for x in cf.mean(axis=(0,1)))}")
print(f"Cycles HDRI-only mean (linear): {tuple(round(float(x),4) for x in ch.mean(axis=(0,1)))}")
print(f"vibrt grid mean (linear): {tuple(round(float(x),4) for x in vg.mean(axis=(0,1)))}")

# Top half (zenith) and bottom half (horizon/ground)
H, _, _ = cf.shape
print()
print("=== top half (theta < π/2, mostly +Z direction) ===")
print(f"Cycles full top: {tuple(round(float(x),4) for x in cf[:H//2].mean(axis=(0,1)))}")
print(f"Cycles HDRI top: {tuple(round(float(x),4) for x in ch[:H//2].mean(axis=(0,1)))}")
print(f"vibrt grid top: {tuple(round(float(x),4) for x in vg[:H//2].mean(axis=(0,1)))}")
print()
print("=== bottom half (theta > π/2, -Z direction = ground) ===")
print(f"Cycles full bottom: {tuple(round(float(x),4) for x in cf[H//2:].mean(axis=(0,1)))}")
print(f"Cycles HDRI bottom: {tuple(round(float(x),4) for x in ch[H//2:].mean(axis=(0,1)))}")
print(f"vibrt grid bottom: {tuple(round(float(x),4) for x in vg[H//2:].mean(axis=(0,1)))}")
