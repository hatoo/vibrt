"""Trigger the sky bake and inspect the resulting envmap pixels — total
energy, peak luminance, mean luminance per region. Lets us decide
whether the residual sky after sun extraction is too dim to deliver
the indirect lighting we expect.
"""
import bpy
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\hato2\Desktop\vibrt\blender")
from vibrt_blender import exporter
from vibrt_blender.export_context import ExportContext
export_context = ExportContext()

w = bpy.context.scene.world
print(f"world={w.name}, use_nodes={w.use_nodes}")
exporter.clear_sky_bake_cache(context=export_context)
exporter.prebake_sky_envmaps_for_world(w, w_=2048 if False else 2048, h=1024, context=export_context) if False else \
    exporter.prebake_sky_envmaps_for_world(w, w=2048, h=1024, context=export_context)

print("\n=== export_context.sky_maps entries ===")
for key, val in export_context.sky_maps.items():
    rgb, bw, bh, sun = val
    rgb_arr = np.asarray(rgb).reshape(bh, bw, 3)
    lum = 0.2126 * rgb_arr[..., 0] + 0.7152 * rgb_arr[..., 1] + 0.0722 * rgb_arr[..., 2]
    print(f"  key={key!r}")
    print(f"    shape={rgb_arr.shape}")
    print(f"    luminance: min={lum.min():.4f} max={lum.max():.4f} mean={lum.mean():.4f} median={np.median(lum):.4f}")
    # Per-region luminance: top (sky), middle (horizon), bottom (ground/mirror).
    h_top, h_mid, h_bot = bh // 4, bh // 2, 3 * bh // 4
    print(f"    top (sky 0-{h_top}): mean={lum[:h_top].mean():.4f} max={lum[:h_top].max():.4f}")
    print(f"    horizon ({h_top}-{h_mid}): mean={lum[h_top:h_mid].mean():.4f} max={lum[h_top:h_mid].max():.4f}")
    print(f"    below horizon ({h_mid}-{h_bot}): mean={lum[h_mid:h_bot].mean():.4f}")
    print(f"    bottom ({h_bot}-{bh}): mean={lum[h_bot:].mean():.4f}")
    # Solid-angle weighted total flux: ∫ L dΩ
    # equirect Jacobian: dΩ = (2π * π / (W*H)) * sin(theta), theta = (y+0.5) * π/H
    import math
    ys = np.arange(bh)
    theta = (ys + 0.5) * (math.pi / bh)
    sin_t = np.sin(theta)[:, None, None]  # (h, 1, 1)
    omega = (2.0 * math.pi * math.pi / (bw * bh)) * sin_t
    total_flux_per_chan = (rgb_arr * omega).sum(axis=(0, 1))
    # Hemispherical irradiance estimate: assume a horizontal surface
    # facing world +Z, then incident irradiance = ∫ L cos(theta) dΩ
    # over the upper hemisphere (theta in [0, π/2]).
    cos_t = np.cos(theta)[:, None, None]  # (h, 1, 1)
    cos_t_clipped = np.where(theta[:, None, None] < math.pi / 2.0, cos_t, 0.0)
    irradiance_horizontal = (rgb_arr * cos_t_clipped * omega).sum(axis=(0, 1))
    print(f"    total flux (∫L dΩ) per channel = {tuple(round(float(c), 4) for c in total_flux_per_chan)}")
    print(f"    horizontal irradiance (∫L cos(θ) dΩ over upper hemisphere) = {tuple(round(float(c), 4) for c in irradiance_horizontal)}")
    if sun is not None:
        print(f"    sun light: {sun}")
        # Sun also contributes irradiance: E_sun_perpendicular × cos(angle from up)
        sun_dir = sun.get("direction", [0, 0, -1])
        cos_sun_up = -float(sun_dir[2])  # direction is "where photons go" so flip
        print(f"    sun cos(angle from +Z) = {cos_sun_up:.3f}")
        sun_str = sun.get("strength", 0.0)
        sun_color = sun.get("color", [1, 1, 1])
        sun_irr = [c * sun_str * max(cos_sun_up, 0.0) for c in sun_color]
        print(f"    sun horizontal irradiance contribution = {tuple(round(c, 4) for c in sun_irr)}")
