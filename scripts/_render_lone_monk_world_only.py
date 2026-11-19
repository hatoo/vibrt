"""Render lone_monk's TexSky world with NO geometry — just a flat
horizontal floor receiving the world's illumination. If cycles and
vibrt deliver similar luminance per pixel here, the gap on the full
scene is in geometry interaction (occlusion / multi-bounce). If they
diverge here, the gap is in direct world sampling itself.
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


def strip_to_floor():
    """Hide every mesh / light except a single flat floor. Floor uses
    a fresh white Diffuse material so cycles + vibrt see the exact
    same surface."""
    # Hide everything renderable
    for obj in bpy.context.scene.objects:
        if obj.type in {"MESH", "LIGHT", "CAMERA"}:
            obj.hide_render = (obj.type != "CAMERA")
    # Add a 200×200 m horizontal plane at z=0
    bpy.ops.mesh.primitive_plane_add(size=200, location=(20, 20, 0))
    floor = bpy.context.object
    floor.name = "__test_floor"
    # Fresh white Diffuse material
    mat = bpy.data.materials.new("__test_floor_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    d = nt.nodes.new("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    d.inputs["Roughness"].default_value = 0.0
    nt.links.new(d.outputs["BSDF"], out.inputs["Surface"])
    if floor.material_slots:
        floor.material_slots[0].material = mat
    else:
        floor.data.materials.append(mat)


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
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "32"
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    # KEY: disable the compositor so we read raw render output instead of
    # running through lone_monk's per-pixel Exposure / Glare / LensDist
    # chain that silently boosts bright pixels.
    scene.render.use_compositing = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


def stats(path):
    img = bpy.data.images.load(str(path))
    img.colorspace_settings.name = "Linear Rec.709"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    arr = flat.reshape((h, w, 4))[..., :3]
    bpy.data.images.remove(img)
    return arr


def main():
    print("=== stripping to floor only ===", flush=True)
    strip_to_floor()
    out_v = HERE / "test_scenes" / "lone_monk" / "_floor_vibrt.exr"
    out_c = HERE / "test_scenes" / "lone_monk" / "_floor_cycles.exr"
    print("=== render vibrt ===", flush=True)
    render("vibrt", 256, 0, out_v, 25)  # db=0: only direct world NEE
    print("=== render cycles ===", flush=True)
    render("cycles", 256, 0, out_c, 25)

    v = stats(out_v)
    c = stats(out_c)
    print(f"\n=== floor-only direct NEE comparison (db=0) ===")
    print(f"  vibrt mean RGB  = ({v[..., 0].mean():.3f}, {v[..., 1].mean():.3f}, {v[..., 2].mean():.3f})")
    print(f"  cycles mean RGB = ({c[..., 0].mean():.3f}, {c[..., 1].mean():.3f}, {c[..., 2].mean():.3f})")
    print(f"  vibrt peak  = {v.max():.2f}, p99 = {np.percentile(v, 99):.2f}, p99.9 = {np.percentile(v, 99.9):.2f}")
    print(f"  cycles peak = {c.max():.2f}, p99 = {np.percentile(c, 99):.2f}, p99.9 = {np.percentile(c, 99.9):.2f}")
    print(f"  total energy ratio v/c = {v.sum() / max(c.sum(), 1e-6):.4f}")


main()
