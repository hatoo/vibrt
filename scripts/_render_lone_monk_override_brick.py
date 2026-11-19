"""Replace 'brick marble smooth' material with constant-color Diffuse and
re-render full lone_monk db=0 compositor=OFF. If the ratio drops at
hot pixel (166, 200), the material's chain (HSV / Mix / TexImage) is
the source of vibrt's over-delivery. Otherwise the issue is elsewhere
(BSDF lobe weighting, texture sampling, etc.).
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


def replace_brick():
    mat = bpy.data.materials.get("brick marble smooth")
    if mat is None:
        print("brick marble smooth not found — skipping override")
        return
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    if out is None:
        return
    for n in list(nt.nodes):
        if n.bl_idname != "ShaderNodeOutputMaterial":
            nt.nodes.remove(n)
    d = nt.nodes.new("ShaderNodeBsdfDiffuse")
    # Use a brick-ish color matching the visual look
    d.inputs["Color"].default_value = (0.4, 0.27, 0.20, 1.0)
    d.inputs["Roughness"].default_value = 0.0
    nt.links.new(d.outputs["BSDF"], out.inputs["Surface"])
    print("[replaced 'brick marble smooth' with Diffuse(0.4, 0.27, 0.20)]", flush=True)


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

    print("=== replace brick ===", flush=True)
    replace_brick()
    out_v = HERE / "test_scenes" / "lone_monk" / "_brick_override_vibrt.exr"
    out_c = HERE / "test_scenes" / "lone_monk" / "_brick_override_cycles.exr"

    print("=== vibrt ===", flush=True)
    configure("vibrt", spp, pct, 0)
    render_to(out_v)
    print("=== cycles ===", flush=True)
    configure("cycles", spp, pct, 0)
    render_to(out_c)

    v = load(out_v); c = load(out_c)
    H, W = min(v.shape[0], c.shape[0]), min(v.shape[1], c.shape[1])
    v, c = v[:H, :W], c[:H, :W]
    print(f"\n=== brick override db=0 (compositor=OFF) ===")
    print(f"  vibrt  mean = ({v[..., 0].mean():.3f}, {v[..., 1].mean():.3f}, {v[..., 2].mean():.3f})  peak={v.max():.2f}")
    print(f"  cycles mean = ({c[..., 0].mean():.3f}, {c[..., 1].mean():.3f}, {c[..., 2].mean():.3f})  peak={c.max():.2f}")
    print(f"  ratio v/c   = {v.sum() / max(c.sum(), 1e-6):.3f}")
    p1 = (166, 200)
    p2 = (185, 142)
    if p1[0] < H and p1[1] < W:
        print(f"  pixel y=166 x=200: vibrt = {tuple(round(float(v[166, 200, k]), 3) for k in range(3))}  cycles = {tuple(round(float(c[166, 200, k]), 3) for k in range(3))}")
    if p2[0] < H and p2[1] < W:
        print(f"  pixel y=185 x=142: vibrt = {tuple(round(float(v[185, 142, k]), 3) for k in range(3))}  cycles = {tuple(round(float(c[185, 142, k]), 3) for k in range(3))}")


main()
