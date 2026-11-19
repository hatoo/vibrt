"""Render the full lone_monk scene (geometry intact, default bounces)
with compositor disabled and view_transform=Raw, in both vibrt and cycles,
and report the linear-space |diff|.

This is the canonical engine-only comparison: previous |diff|=0.150 included
lone_monk's CompositorNodeExposure chain which amplified bright pixels
unequally between the two engines.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "blender"))

import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass


def configure(engine: str, spp: int, percentage: int):
    scene = bpy.context.scene
    scene.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        scene.vibrt_spp = spp
    else:
        scene.cycles.samples = spp
        scene.cycles.use_adaptive_sampling = False
    scene.render.resolution_percentage = percentage
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "32"
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.use_compositing = False


def render_to(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


def load_exr(path: Path):
    img = bpy.data.images.load(str(path))
    img.colorspace_settings.name = "Linear Rec.709"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    arr = flat.reshape((h, w, 4))[..., :3]
    bpy.data.images.remove(img)
    return arr


def main():
    spp = int(os.environ.get("SPP", "128"))
    pct = int(os.environ.get("PCT", "50"))
    out_v = HERE / "test_scenes" / "lone_monk" / "_full_nocomposite_vibrt.exr"
    out_c = HERE / "test_scenes" / "lone_monk" / "_full_nocomposite_cycles.exr"

    print(f"=== full lone_monk render @ spp={spp}, pct={pct}, compositor=OFF, view=Raw ===", flush=True)
    print("=== render vibrt ===", flush=True)
    configure("vibrt", spp, pct)
    render_to(out_v)
    print("=== render cycles ===", flush=True)
    configure("cycles", spp, pct)
    render_to(out_c)

    v = load_exr(out_v)
    c = load_exr(out_c)

    # Match vibrt to cycles dims (should already match)
    if v.shape != c.shape:
        print(f"!! shape mismatch: vibrt {v.shape} vs cycles {c.shape} — using crop")
        h = min(v.shape[0], c.shape[0])
        w = min(v.shape[1], c.shape[1])
        v = v[:h, :w]
        c = c[:h, :w]

    diff = v - c
    abs_diff = np.abs(diff)

    print(f"\n=== linear-space comparison (compositor off) ===")
    print(f"  vibrt  mean RGB = ({v[..., 0].mean():.3f}, {v[..., 1].mean():.3f}, {v[..., 2].mean():.3f})")
    print(f"  cycles mean RGB = ({c[..., 0].mean():.3f}, {c[..., 1].mean():.3f}, {c[..., 2].mean():.3f})")
    print(f"  diff   mean RGB = ({diff[..., 0].mean():+.3f}, {diff[..., 1].mean():+.3f}, {diff[..., 2].mean():+.3f})")
    print(f"  |diff| mean RGB = ({abs_diff[..., 0].mean():.4f}, {abs_diff[..., 1].mean():.4f}, {abs_diff[..., 2].mean():.4f})  overall={abs_diff.mean():.4f}")
    print(f"  vibrt  peak  = {v.max():.2f}, p99 = {np.percentile(v, 99):.2f}")
    print(f"  cycles peak  = {c.max():.2f}, p99 = {np.percentile(c, 99):.2f}")
    print(f"  total energy ratio v/c = {v.sum() / max(c.sum(), 1e-6):.4f}")

    # Compare with previous PNG-based |diff|=0.150 baseline (which went through
    # both compositor and Filmic view transform — apples to oranges, but
    # gives the order-of-magnitude). PNG values are sRGB-encoded 0..1 floats.
    # Here we report linear-space |diff| only.
    # If linear |diff| << 0.150, the metric was compositor/tonemap-dominated.


main()
