"""Lone_monk all-white materials, db=0, compositor=OFF — verifies whether
vibrt's full-scene over-delivery vanishes when materials are uniform white
Diffuse (matching the floor-only test BSDF). If yes → real-material BSDF
eval is the cause. If no → geometry/scene structure is the cause.
"""
from __future__ import annotations
import os, sys
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


def whitify():
    for mat in bpy.data.materials:
        if mat.node_tree is None:
            continue
        nt = mat.node_tree
        out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
        if out is None:
            continue
        for n in list(nt.nodes):
            if n.bl_idname != "ShaderNodeOutputMaterial":
                nt.nodes.remove(n)
        d = nt.nodes.new("ShaderNodeBsdfDiffuse")
        d.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        d.inputs["Roughness"].default_value = 0.0
        nt.links.new(d.outputs["BSDF"], out.inputs["Surface"])


def configure(engine, spp, pct, db):
    s = bpy.context.scene
    s.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        s.vibrt_spp = spp
    else:
        s.cycles.samples = spp
        s.cycles.use_adaptive_sampling = False
    s.cycles.diffuse_bounces = db
    s.render.resolution_percentage = pct
    s.render.image_settings.file_format = "OPEN_EXR"
    s.render.image_settings.color_mode = "RGB"
    s.render.image_settings.color_depth = "32"
    s.view_settings.view_transform = "Raw"
    s.view_settings.look = "None"
    s.view_settings.exposure = 0.0
    s.view_settings.gamma = 1.0
    s.render.use_compositing = False


def render_to(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)


def load(p):
    img = bpy.data.images.load(str(p))
    img.colorspace_settings.name = "Linear Rec.709"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    arr = flat.reshape((h, w, 4))[..., :3]
    bpy.data.images.remove(img)
    return arr


def main():
    spp = int(os.environ.get("SPP", "64"))
    pct = int(os.environ.get("PCT", "25"))
    print("=== whitify all materials ===", flush=True)
    whitify()
    out_v = HERE / "test_scenes" / "lone_monk" / "_white_db0_vibrt.exr"
    out_c = HERE / "test_scenes" / "lone_monk" / "_white_db0_cycles.exr"
    print("=== vibrt db=0 ===", flush=True)
    configure("vibrt", spp, pct, 0)
    render_to(out_v)
    print("=== cycles db=0 ===", flush=True)
    configure("cycles", spp, pct, 0)
    render_to(out_c)
    v = load(out_v); c = load(out_c)
    H, W = min(v.shape[0], c.shape[0]), min(v.shape[1], c.shape[1])
    v, c = v[:H, :W], c[:H, :W]
    print(f"\n=== all-white db=0 ===")
    print(f"  vibrt  mean = ({v[..., 0].mean():.3f}, {v[..., 1].mean():.3f}, {v[..., 2].mean():.3f})  peak={v.max():.2f}")
    print(f"  cycles mean = ({c[..., 0].mean():.3f}, {c[..., 1].mean():.3f}, {c[..., 2].mean():.3f})  peak={c.max():.2f}")
    print(f"  ratio v/c   = {v.sum() / max(c.sum(), 1e-6):.3f}")
    print(f"  pixel y=166 x=200: vibrt={tuple(round(float(v[166, 200, k]), 3) for k in range(3))}  cycles={tuple(round(float(c[166, 200, k]), 3) for k in range(3))}")
    print(f"  pixel y=185 x=142: vibrt={tuple(round(float(v[min(185, H-1), min(142, W-1), k]), 3) for k in range(3))}  cycles={tuple(round(float(c[min(185, H-1), min(142, W-1), k]), 3) for k in range(3))}")


main()
