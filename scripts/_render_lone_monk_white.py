"""All-white materials debug render. Force every Principled material's
base_color to (0.5, 0.5, 0.5), strip textures from base_color/roughness/
metallic, and zero out emission everywhere except the world. Render
lone_monk with both vibrt and Cycles. Comparing the two outputs
isolates whether the |diff|=0.150 gap is in (a) material textures /
shaders or (b) the lighting / path tracing core.

Run via:
  blender -b test_scenes/lone_monk/lone_monk/lone_monk.blend \
      --python scripts/_render_lone_monk_white.py
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


def whitify_materials():
    """Strip every material to a flat grey Principled BSDF. Idempotent."""
    for mat in bpy.data.materials:
        if mat.node_tree is None:
            continue
        nt = mat.node_tree
        # Find the OUTPUT and rebuild a minimal graph: white Principled.
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if out is None:
            continue
        # Save existing surface link and clear everything else.
        for n in list(nt.nodes):
            if n.type != "OUTPUT_MATERIAL":
                nt.nodes.remove(n)
        new_principled = nt.nodes.new("ShaderNodeBsdfPrincipled")
        new_principled.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        new_principled.inputs["Roughness"].default_value = 0.5
        new_principled.inputs["Metallic"].default_value = 0.0
        try:
            new_principled.inputs["Specular IOR Level"].default_value = 0.5
        except KeyError:
            pass
        nt.links.new(new_principled.outputs["BSDF"], out.inputs["Surface"])


def render(engine, spp, out_path, percentage=50):
    scene = bpy.context.scene
    scene.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        scene.vibrt_spp = spp
    else:
        scene.cycles.samples = spp
        scene.cycles.use_adaptive_sampling = False
    scene.render.resolution_percentage = percentage
    scene.render.image_settings.file_format = "PNG"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


def stats(path):
    img = bpy.data.images.load(str(path))
    img.colorspace_settings.name = "sRGB"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    arr = flat.reshape((h, w, 4))[..., :3]
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    p25 = np.percentile(lum, 25)
    p75 = np.percentile(lum, 75)
    out = {}
    for label, mask in (("all", np.ones(arr.shape[:2], dtype=bool)),
                         ("dark25", lum < p25),
                         ("mid50", (lum >= p25) & (lum < p75)),
                         ("bright25", lum >= p75)):
        if mask.sum() > 0:
            mean = arr[mask].mean(axis=0)
            out[label] = tuple(round(float(c), 4) for c in mean)
        else:
            out[label] = (0, 0, 0)
    bpy.data.images.remove(img)
    return out


def main():
    print("=== whitifying all materials ===", flush=True)
    whitify_materials()
    print(f"=== rendering at SPP=128, PCT=50 ===", flush=True)
    rows = []
    for engine in ("vibrt", "cycles"):
        out = HERE / "test_scenes" / "lone_monk" / f"_white_{engine}.png"
        print(f"\n--- {engine} ---", flush=True)
        render(engine, 128, out, 50)
        s = stats(out)
        rows.append((engine, s))
        for k, v in s.items():
            print(f"  {k}: {v}", flush=True)
    print("\n=== summary ===")
    for engine, s in rows:
        print(f"  {engine:6s} all={s['all']} dark={s['dark25']} bright={s['bright25']}")
    if len(rows) == 2:
        v_all, c_all = rows[0][1]['all'], rows[1][1]['all']
        diff = tuple(round(v - c, 4) for v, c in zip(v_all, c_all))
        print(f"\n  all signed diff (vibrt - cycles): {diff}")


main()
