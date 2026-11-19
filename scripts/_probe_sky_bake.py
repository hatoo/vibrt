"""Run vibrt's sky bake on the loaded scene's world and dump statistics
+ a small downscaled EXR/PNG so we can eyeball the sky colour vs Cycles.

Run via: blender -b <scene>.blend --python scripts/_probe_sky_bake.py
"""

import os
import sys
import math
import bpy
import numpy as np

# Make the addon importable directly from the working tree.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))

import vibrt_blender.exporter as exporter
import vibrt_blender.material_export as material_export

scene = bpy.context.scene
world = scene.world
print(f"world: {world.name}")

# Bake via the exact path the addon uses.
exporter.clear_sky_bake_cache()
exporter.prebake_sky_envmaps_for_world(world)

# Pull the cached bake back out.
sky_node = next(
    (n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeTexSky"),
    None,
)
key = exporter._sky_node_cache_key(world, sky_node)
baked = exporter._SKY_BAKE_CACHE.get(key)
if baked is None:
    print("ERROR: bake cache miss")
    sys.exit(1)

rgb, w, h, sun_light = baked
print(f"bake size:   {w} x {h}")
print(f"bake dtype:  {rgb.dtype}, shape={rgb.shape}")

# Bake stats by hemisphere and over the whole sphere.
ys = np.arange(h, dtype=np.float32)
theta = (ys + 0.5) * (math.pi / h)         # 0 = +Z (zenith)
sin_t = np.sin(theta)
omega_row = (2.0 * math.pi * math.pi / (w * h)) * sin_t   # per-pixel sr at row y
weight = np.broadcast_to(omega_row[:, None, None], rgb.shape)

# Total integrated radiance (irradiance ish) over the upper hemisphere.
upper_mask = (theta < math.pi / 2.0)
cos_t = np.cos(theta).astype(np.float32)
# E = ∫ L cos θ dω  over upper hemisphere
upper_w = (np.maximum(cos_t, 0.0) * (2.0 * math.pi * math.pi / (w * h)) * sin_t)[:, None, None]
upper_w_b = np.broadcast_to(upper_w, rgb.shape)
E_rgb = (rgb * upper_w_b).sum(axis=(0, 1))
print(f"upper-hemisphere irradiance E_rgb (sr-integrated cos-weighted, linear):")
print(f"  R = {E_rgb[0]:.4f}")
print(f"  G = {E_rgb[1]:.4f}")
print(f"  B = {E_rgb[2]:.4f}")
print(f"  total luminance = {0.2126*E_rgb[0] + 0.7152*E_rgb[1] + 0.0722*E_rgb[2]:.4f}")
print(f"  R/G = {E_rgb[0]/max(E_rgb[1], 1e-9):.4f}")
print(f"  B/G = {E_rgb[2]/max(E_rgb[1], 1e-9):.4f}")

# Per-channel max & mean
print()
print(f"raw per-channel (no weighting):")
print(f"  mean = ({rgb[..., 0].mean():.4f}, {rgb[..., 1].mean():.4f}, {rgb[..., 2].mean():.4f})")
print(f"  max  = ({rgb[..., 0].max():.4f}, {rgb[..., 1].max():.4f}, {rgb[..., 2].max():.4f})")

# A handful of fixed sample directions, expressed in the same convention
# as the bake (y=0 at zenith, x sweeps phi).
def sample(theta_deg, phi_deg):
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg) % (2 * math.pi)
    py = int(min(h - 1, max(0, math.floor(th / math.pi * h))))
    px = int(min(w - 1, max(0, math.floor(ph / (2 * math.pi) * w))))
    return rgb[py, px, :]

print()
print(f"sample directions (theta from zenith, phi from +X CCW):")
for tdeg in (5, 30, 60, 89):
    for pdeg in (0, 90, 180, 270):
        c = sample(tdeg, pdeg)
        lum = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
        print(f"  θ={tdeg:>3}° φ={pdeg:>3}°: rgb=({c[0]:7.3f}, {c[1]:7.3f}, {c[2]:7.3f}) "
              f"lum={lum:.3f} R/B={c[0]/max(c[2],1e-6):.3f}")

# Sun split.
print()
print("sun split:")
if sun_light is None:
    print("  (no sun extracted)")
else:
    print(f"  type={sun_light['type']}")
    print(f"  direction (travel) = {sun_light['direction']}")
    print(f"  color  = {sun_light['color']}")
    print(f"  strength = {sun_light['strength']}")
    print(f"  angle_rad = {sun_light['angle_rad']} ({math.degrees(sun_light['angle_rad']):.2f}°)")
    sd = sun_light['direction']
    sc = sun_light['color']
    print(f"  R/G = {sc[0]/max(sc[1], 1e-9):.4f}")
    print(f"  B/G = {sc[2]/max(sc[1], 1e-9):.4f}")
    # Effective irradiance on a horizontal plane (sun comes from -direction).
    cos_inc = max(0.0, -sd[2])
    print(f"  cos(incidence on horizontal) = {cos_inc:.4f}")
    irr_rgb = [sc[0]*sun_light['strength']*cos_inc,
               sc[1]*sun_light['strength']*cos_inc,
               sc[2]*sun_light['strength']*cos_inc]
    print(f"  horizontal irradiance R/G/B = {irr_rgb}")

# Save a small downsampled PNG (Filmic exposure -2 like the scene) so we
# can eyeball the bake.
out_path = os.path.join(HERE, "test_scenes", "lone_monk", "preview_skybake.exr")
try:
    img = bpy.data.images.new("__sky_dump", width=w, height=h, alpha=False, float_buffer=True)
    flat = np.empty((h, w, 4), dtype=np.float32)
    flat[..., :3] = rgb
    flat[..., 3] = 1.0
    img.pixels.foreach_set(flat.reshape(-1))
    img.filepath_raw = out_path
    img.file_format = "OPEN_EXR"
    img.save()
    print(f"\nwrote bake → {out_path}")
except Exception as e:
    print(f"\nfailed to write bake: {e}")
