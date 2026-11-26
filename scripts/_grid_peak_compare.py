"""Compare vibrt's Mixed grid peak luminance and pixel distribution vs
Cycles' aligned full-world equirect. Hot HDRI pixels in the grid are
what the CDF importance-samples; over-bright outliers in the grid will
make NEE over-contribute on the sky direction and dim everything else."""
import bpy, numpy as np, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_lin(path):
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = "Non-Color"
    arr = np.empty(img.size[0]*img.size[1]*4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    rgb = arr.reshape((img.size[1], img.size[0], 4))[::-1, :, :3]
    # PNGs from Blender with view_transform=Standard are sRGB-encoded.
    return np.where(rgb <= 0.04045, rgb/12.92, ((rgb+0.055)/1.055)**2.4)

# vibrt grid (PNG saved by my dump script with Reinhard tone-map; that's
# in display space, not linear radiance).
# Re-rasterise the grid in Python to get LINEAR values matching what the
# kernel sees.
import sys, math
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
from vibrt_blender.export_context import ExportContext
export_context = ExportContext()
try: bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception: pass
from vibrt_blender.exporter import prebake_sky_envmaps_for_world, _sky_node_alone_cache_key, _detect_mix_world, _euler_xyz_to_matrix3

scene = bpy.context.scene
prebake_sky_envmaps_for_world(scene.world, w=1024, h=512, context=export_context)

mix = _detect_mix_world(scene.world)
bg_a, bg_b, fac = mix

def get_layer(bg):
    src = bg.inputs["Color"].links[0].from_node
    strength = float(bg.inputs["Strength"].default_value)
    if src.bl_idname == "ShaderNodeTexSky":
        rgb_b, w, h, _ = export_context.sky_maps[_sky_node_alone_cache_key(scene.world, src)]
        rgb = (rgb_b if isinstance(rgb_b, np.ndarray)
               else np.frombuffer(rgb_b, dtype=np.float32)).reshape((h, w, 3))
        return ("equirect", rgb, w, h, np.eye(3), strength, "repeat")
    img = src.image
    prev = img.colorspace_settings.name
    img.colorspace_settings.name = "Non-Color"
    try:
        arr = np.empty(img.size[0]*img.size[1]*4, dtype=np.float32)
        img.pixels.foreach_get(arr)
    finally:
        img.colorspace_settings.name = prev
    rgba = arr.reshape((img.size[1], img.size[0], 4))[..., :3]
    if prev.lower().startswith("srgb"):
        rgba = np.where(rgba <= 0.04045, rgba/12.92, ((rgba+0.055)/1.055)**2.4)
    rot = np.eye(3)
    v = src.inputs.get("Vector")
    if v and v.is_linked:
        mp = v.links[0].from_node
        if mp.bl_idname == "ShaderNodeMapping":
            rx, ry, rz = mp.inputs["Rotation"].default_value
            rot = np.array(_euler_xyz_to_matrix3(rx, ry, rz)).reshape(3,3)
    proj = "flat" if src.projection == "FLAT" else "equirect"
    ext = {"REPEAT": "repeat", "EXTEND": "extend"}.get(src.extension, "repeat")
    return (proj, rgba, img.size[0], img.size[1], rot, strength, ext)

la = get_layer(bg_a); lb = get_layer(bg_b)

def sample(layer, dirs):
    proj, rgb, w, h, rot, strength, ext = layer
    rotated = dirs @ rot.T
    if proj == "flat":
        u, v = rotated[:, 0], rotated[:, 1]
    else:
        z = np.clip(rotated[:, 2], -1, 1)
        u = np.arctan2(rotated[:, 1], rotated[:, 0]) * (0.5/math.pi)
        u = np.where(u < 0, u + 1, u)
        v = np.arccos(z) / math.pi
    fx = (u % 1.0) * w; fy = (v % 1.0) * h
    xi = np.clip(fx.astype(int), 0, w - 1); yi = np.clip(fy.astype(int), 0, h - 1)
    return rgb[yi, xi, :] * strength

target_w, target_h = 2048, 1024
us = (np.arange(target_w) + 0.5) / target_w
vs = (np.arange(target_h) + 0.5) / target_h
uu, vv = np.meshgrid(us, vs)
phi = uu * 2 * math.pi; theta = vv * math.pi
dirs = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=-1).reshape(-1, 3)
a = sample(la, dirs); b = sample(lb, dirs)
mixed = (a * (1 - fac) + b * fac).reshape(target_h, target_w, 3)

# Solid-angle weighted integral (= total flux from the envmap)
sin_t = np.sin((np.arange(target_h) + 0.5) * (math.pi / target_h))[:, None, None]
solid = (2 * math.pi * math.pi / (target_w * target_h)) * sin_t
flux_per_channel = (mixed * solid).sum(axis=(0, 1))
print(f"vibrt mixed grid: shape={mixed.shape}")
print(f"  mean rgb (linear, unweighted): {tuple(round(float(x),3) for x in mixed.mean(axis=(0,1)))}")
print(f"  flux ∫L·dω (linear): {tuple(round(float(x),3) for x in flux_per_channel)}")
print(f"  max value: {float(mixed.max()):.2f}")
lum = 0.2126*mixed[...,0] + 0.7152*mixed[...,1] + 0.0722*mixed[...,2]
print(f"  luminance: max={float(lum.max()):.2f}  p99.9={float(np.percentile(lum, 99.9)):.2f}  p50={float(np.percentile(lum, 50)):.2f}")
print(f"  hot pixel cluster (>p99.9): {int((lum > np.percentile(lum, 99.9)).sum())} pixels")

# Cycles aligned full eq for comparison
cycles_eq = load_lin(os.path.join(HERE, "test_scenes", "pabellon", "_cycles_equirect_world.png"))
print()
print(f"Cycles aligned full eq: shape={cycles_eq.shape}")
print(f"  mean rgb (linear): {tuple(round(float(x),3) for x in cycles_eq.mean(axis=(0,1)))}")
sin_tc = np.sin((np.arange(cycles_eq.shape[0]) + 0.5) * (math.pi / cycles_eq.shape[0]))[:, None, None]
solid_c = (2 * math.pi * math.pi / (cycles_eq.shape[0] * cycles_eq.shape[1])) * sin_tc
flux_c = (cycles_eq * solid_c).sum(axis=(0, 1))
print(f"  flux: {tuple(round(float(x),3) for x in flux_c)}")
print(f"  max value: {float(cycles_eq.max()):.4f}")
lum_c = 0.2126*cycles_eq[...,0] + 0.7152*cycles_eq[...,1] + 0.0722*cycles_eq[...,2]
print(f"  luminance: max={float(lum_c.max()):.4f}  p99.9={float(np.percentile(lum_c, 99.9)):.4f}  p50={float(np.percentile(lum_c, 50)):.4f}")
