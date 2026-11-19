"""Render full lone_monk db=0 with all emissive objects hidden, to isolate
whether vibrt's 60% over-delivery vs cycles is from mesh-light NEE on
emissive surfaces, or from sun/envmap NEE on regular surfaces.
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


def has_emission(obj):
    """Mesh has any material with non-zero EFFECTIVE emission (color * strength > 0)?
    Returns True if there is an Emission shader or a Principled with both
    color != black (or linked) AND strength > 0 (or linked).
    """
    if obj.type != "MESH":
        return False
    for ms in obj.material_slots:
        m = ms.material
        if m is None or not m.use_nodes:
            continue
        for n in m.node_tree.nodes:
            if n.bl_idname == "ShaderNodeEmission":
                col = n.inputs.get("Color")
                s = n.inputs.get("Strength")
                col_nonzero = col is not None and (col.is_linked or any(float(x) > 0 for x in col.default_value[:3]))
                str_nonzero = s is not None and (s.is_linked or float(s.default_value) > 0)
                if col_nonzero and str_nonzero:
                    return True
            elif n.bl_idname == "ShaderNodeBsdfPrincipled":
                col = n.inputs.get("Emission Color")
                s = n.inputs.get("Emission Strength")
                col_nonzero = col is not None and (col.is_linked or any(float(x) > 0 for x in col.default_value[:3]))
                str_nonzero = s is not None and (s.is_linked or float(s.default_value) > 0)
                if col_nonzero and str_nonzero:
                    return True
    return False


def hide_emitters():
    n = 0
    for obj in bpy.context.scene.objects:
        if has_emission(obj):
            obj.hide_render = True
            n += 1
    print(f"[hidden {n} emissive objects]", flush=True)
    return n


def configure(engine: str, spp: int, percentage: int, db: int):
    scene = bpy.context.scene
    scene.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        scene.vibrt_spp = spp
    else:
        scene.cycles.samples = spp
        scene.cycles.use_adaptive_sampling = False
    scene.cycles.diffuse_bounces = db
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
    spp = int(os.environ.get("SPP", "64"))
    pct = int(os.environ.get("PCT", "25"))

    n = hide_emitters()
    out_v = HERE / "test_scenes" / "lone_monk" / "_noemit_db0_vibrt.exr"
    out_c = HERE / "test_scenes" / "lone_monk" / "_noemit_db0_cycles.exr"

    print(f"=== vibrt db=0 (no emitters) ===", flush=True)
    configure("vibrt", spp, pct, 0)
    render_to(out_v)
    print(f"=== cycles db=0 (no emitters) ===", flush=True)
    configure("cycles", spp, pct, 0)
    render_to(out_c)

    v = load_exr(out_v)
    c = load_exr(out_c)
    H = min(v.shape[0], c.shape[0]); W = min(v.shape[1], c.shape[1])
    v, c = v[:H, :W], c[:H, :W]
    print(f"\n=== db=0 no-emitters comparison ===")
    print(f"  vibrt  mean = ({v[..., 0].mean():.3f}, {v[..., 1].mean():.3f}, {v[..., 2].mean():.3f})  peak={v.max():.2f}")
    print(f"  cycles mean = ({c[..., 0].mean():.3f}, {c[..., 1].mean():.3f}, {c[..., 2].mean():.3f})  peak={c.max():.2f}")
    print(f"  ratio v/c = {v.sum() / max(c.sum(), 1e-6):.3f}")

    # Same hot region check
    print(f"\n  pixel y=166 x=200: vibrt = {tuple(round(float(v[166, 200, k]), 2) for k in range(3))}  cycles = {tuple(round(float(c[166, 200, k]), 2) for k in range(3))}")
    print(f"  pixel y=185 x=142: vibrt = {tuple(round(float(v[min(185, H-1), min(142, W-1), k]), 2) for k in range(3))}  cycles = {tuple(round(float(c[min(185, H-1), min(142, W-1), k]), 2) for k in range(3))}")


main()
