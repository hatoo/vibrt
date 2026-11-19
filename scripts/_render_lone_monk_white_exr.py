"""All-white materials lone_monk render → linear EXR (no display
transform). Reads back as numpy and computes per-pixel ratio in true
linear space. Filmic-corrected via raw view_transform = 'Raw'."""
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


def whitify_materials():
    for mat in bpy.data.materials:
        if mat.node_tree is None:
            continue
        nt = mat.node_tree
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if out is None:
            continue
        for n in list(nt.nodes):
            if n.type != "OUTPUT_MATERIAL":
                nt.nodes.remove(n)
        d = nt.nodes.new("ShaderNodeBsdfDiffuse")
        d.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        d.inputs["Roughness"].default_value = 0.0
        nt.links.new(d.outputs["BSDF"], out.inputs["Surface"])


def render(engine, spp, db, out_path, percentage=50):
    scene = bpy.context.scene
    scene.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        scene.vibrt_spp = spp
    else:
        scene.cycles.samples = spp
        scene.cycles.use_adaptive_sampling = False
    scene.cycles.diffuse_bounces = db
    scene.render.resolution_percentage = percentage
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '32'
    # Raw view transform → linear-light pixels written verbatim.
    scene.view_settings.view_transform = 'Raw'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


def linear_stats(path):
    """Read linear-light EXR via Blender's image API."""
    img = bpy.data.images.load(str(path))
    img.colorspace_settings.name = "Linear Rec.709"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    arr = flat.reshape((h, w, 4))[..., :3]
    bpy.data.images.remove(img)
    return arr


def main():
    print("=== whitifying ===", flush=True)
    whitify_materials()
    out_v = HERE / "test_scenes" / "lone_monk" / "_white_vibrt.exr"
    out_c = HERE / "test_scenes" / "lone_monk" / "_white_cycles.exr"
    print("=== render vibrt ===", flush=True)
    render("vibrt", 128, 2, out_v, 50)
    print("=== render cycles ===", flush=True)
    render("cycles", 128, 2, out_c, 50)

    v = linear_stats(out_v)
    c = linear_stats(out_c)
    print(f"\n=== linear stats ===")
    print(f"  vibrt:  mean RGB = ({v[..., 0].mean():.4f}, {v[..., 1].mean():.4f}, {v[..., 2].mean():.4f})")
    print(f"  cycles: mean RGB = ({c[..., 0].mean():.4f}, {c[..., 1].mean():.4f}, {c[..., 2].mean():.4f})")
    print(f"  vibrt:  peak = {v.max():.3f}, p99 = {np.percentile(v, 99):.3f}")
    print(f"  cycles: peak = {c.max():.3f}, p99 = {np.percentile(c, 99):.3f}")
    print(f"  total energy ratio v/c = {v.sum() / c.sum():.4f}")

    # Bucket by cycles luminance
    clum = 0.2126*c[...,0] + 0.7152*c[...,1] + 0.0722*c[...,2]
    vlum = 0.2126*v[...,0] + 0.7152*v[...,1] + 0.0722*v[...,2]
    buckets = [(0, 0.01, 'dark'), (0.01, 0.1, 'mid_low'), (0.1, 1, 'mid'),
               (1, 5, 'bright'), (5, 50, 'super_bright'), (50, 1e9, 'sun')]
    for lo, hi, name in buckets:
        mask = (clum >= lo) & (clum < hi)
        n = int(mask.sum())
        if n < 100:
            continue
        v_sum = float(v[mask].sum())
        c_sum = float(c[mask].sum())
        v_mean = float(vlum[mask].mean())
        c_mean = float(clum[mask].mean())
        print(f"  {name:12s} ({lo:>5.2f}-{hi:>5.2f}): n={n:>6d}, "
              f"vmean={v_mean:.4f}, cmean={c_mean:.4f}, "
              f"vsum={v_sum:.0f}, csum={c_sum:.0f}, "
              f"v/c={v_sum/c_sum:.3f}")


main()
