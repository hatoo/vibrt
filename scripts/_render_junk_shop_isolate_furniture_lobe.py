"""Isolate which Furniture lobe (SSS or Transmission) causes the 1000x bug.
Run 4 variants:
  A: original (baseline)
  B: SSS=0 (kill subsurface, keep transmission)
  C: Transmission=0 (kill transmission, keep SSS)
  D: both 0 (kill both)
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


def patch_furniture(disable_sss: bool, disable_transmission: bool):
    """Modify the Principled BSDF on Furniture material in place."""
    mat = bpy.data.materials.get("Furniture")
    if mat is None:
        return
    nt = mat.node_tree
    p = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if p is None:
        print("no Principled BSDF found")
        return
    if disable_sss:
        sw = p.inputs.get("Subsurface Weight")
        if sw is not None:
            if sw.is_linked:
                # remove link
                for link in list(nt.links):
                    if link.to_socket == sw:
                        nt.links.remove(link)
            sw.default_value = 0.0
        # Also unlink Subsurface Scale just in case
        ss = p.inputs.get("Subsurface Scale")
        if ss is not None and ss.is_linked:
            for link in list(nt.links):
                if link.to_socket == ss:
                    nt.links.remove(link)
            ss.default_value = 0.0
    if disable_transmission:
        tw = p.inputs.get("Transmission Weight")
        if tw is not None:
            if tw.is_linked:
                for link in list(nt.links):
                    if link.to_socket == tw:
                        nt.links.remove(link)
            tw.default_value = 0.0
    print(f"[patched Furniture: SSS={'OFF' if disable_sss else 'ON'}, Trans={'OFF' if disable_transmission else 'ON'}]", flush=True)


def configure(engine, spp, pct):
    s = bpy.context.scene
    s.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        s.vibrt_spp = spp
    else:
        s.cycles.samples = spp
        s.cycles.use_adaptive_sampling = False
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
    variant = os.environ.get("VAR", "B")  # default: SSS=OFF, Trans=ON
    if variant == "A":
        pass  # original
    elif variant == "B":
        patch_furniture(True, False)  # SSS off, Trans on
    elif variant == "C":
        patch_furniture(False, True)  # Trans off, SSS on
    elif variant == "D":
        patch_furniture(True, True)
    else:
        raise SystemExit(f"unknown VAR={variant}")

    out_v = HERE / "test_scenes" / "junk_shop" / f"_lobe_{variant}_vibrt.exr"
    out_c = HERE / "test_scenes" / "junk_shop" / f"_lobe_{variant}_cycles.exr"
    print(f"=== variant {variant} vibrt ===", flush=True)
    configure("vibrt", spp, pct)
    render_to(out_v)
    print(f"=== variant {variant} cycles ===", flush=True)
    configure("cycles", spp, pct)
    render_to(out_c)
    v = load(out_v); c = load(out_c)
    H, W = min(v.shape[0], c.shape[0]), min(v.shape[1], c.shape[1])
    v, c = v[:H, :W], c[:H, :W]
    print(f"\n=== junk_shop variant {variant} ===")
    print(f"  vibrt  mean = ({v[..., 0].mean():.3f}, {v[..., 1].mean():.3f}, {v[..., 2].mean():.3f})  peak={v.max():.2f}")
    print(f"  cycles mean = ({c[..., 0].mean():.3f}, {c[..., 1].mean():.3f}, {c[..., 2].mean():.3f})  peak={c.max():.2f}")
    print(f"  ratio v/c = {v.sum() / max(c.sum(), 1e-6):.3f}")
    if H > 109 and W > 169:
        print(f"  pixel y=109 x=169: vibrt = {tuple(round(float(v[109, 169, k]), 3) for k in range(3))}  cycles = {tuple(round(float(c[109, 169, k]), 3) for k in range(3))}")


main()
