"""All-white materials × varying diffuse_bounces. Tells us how the
geometric bounce expansion differs between vibrt and Cycles."""
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
        new_p = nt.nodes.new("ShaderNodeBsdfPrincipled")
        new_p.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        new_p.inputs["Roughness"].default_value = 0.5
        new_p.inputs["Metallic"].default_value = 0.0
        try:
            new_p.inputs["Specular IOR Level"].default_value = 0.0
        except KeyError:
            pass
        nt.links.new(new_p.outputs["BSDF"], out.inputs["Surface"])


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
    mean = arr.mean(axis=(0, 1))
    bpy.data.images.remove(img)
    return tuple(round(float(c), 4) for c in mean)


def main():
    print("=== whitifying ===", flush=True)
    whitify_materials()
    rows = []
    for engine in ("vibrt", "cycles"):
        for db in (0, 1, 2):
            out = HERE / "test_scenes" / "lone_monk" / f"_white_{engine}_db{db}.png"
            print(f"\n--- {engine} db={db} ---", flush=True)
            render(engine, 128, db, out, percentage=50)
            mean = stats(out)
            rows.append((engine, db, mean))
            print(f"  mean: {mean}", flush=True)
    print("\n=== summary (mean RGB) ===")
    for engine, db, mean in rows:
        print(f"  {engine:6s} db={db}: {mean}")


main()
