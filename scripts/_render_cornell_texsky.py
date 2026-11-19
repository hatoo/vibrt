"""Build a fresh Blender scene from scratch with:
  - 200×200 m white Diffuse floor
  - TexSky world (Nishita single-scattering, sun_disc=True,
    matching lone_monk's settings)
  - Camera 1.0 m above floor, looking forward + slightly down
Render with both engines and compare. Goal: see whether the cycles
'extra delivery' anomaly observed on lone_monk is reproducible on a
totally clean scene with no extra geometry.

Run via:
  blender -b --factory-startup --python scripts/_render_cornell_texsky.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import bpy
import numpy as np
import math

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "blender"))

import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass


def build_scene():
    # Clear default scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    # White floor 200×200 m at z=0
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "floor"
    mat = bpy.data.materials.new("floor_white")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    d = nt.nodes.new("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    d.inputs["Roughness"].default_value = 0.0
    nt.links.new(d.outputs["BSDF"], out.inputs["Surface"])
    if not floor.material_slots:
        floor.data.materials.append(mat)
    else:
        floor.material_slots[0].material = mat

    # Camera at z=1m looking forward and slightly down
    bpy.ops.object.camera_add(location=(0, -10, 2))
    cam = bpy.context.object
    cam.rotation_euler = (math.radians(80), 0, 0)
    bpy.context.scene.camera = cam

    # World with Nishita single-scattering TexSky, sun_disc=True,
    # same params as lone_monk
    world = bpy.data.worlds.new("test_world")
    bpy.context.scene.world = world
    world.use_nodes = True
    wnt = world.node_tree
    for n in list(wnt.nodes):
        wnt.nodes.remove(n)
    wout = wnt.nodes.new("ShaderNodeOutputWorld")
    bg = wnt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.0
    sky = wnt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = 'SINGLE_SCATTERING'
    sky.sun_disc = True
    sky.sun_size = 0.01745329  # 1° in radians
    sky.sun_intensity = 1.0
    sky.sun_elevation = 0.925   # ~53°
    sky.sun_rotation = 2.618    # ~150°
    sky.altitude = 0.0
    sky.air_density = 1.0
    try:
        sky.aerosol_density = 1.0
    except AttributeError:
        sky.dust_density = 1.0
    sky.ozone_density = 1.0
    sky.ground_albedo = 0.3
    sky.turbidity = 2.2
    wnt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    wnt.links.new(bg.outputs["Background"], wout.inputs["Surface"])


def render(engine, spp, db, out_path, percentage=50):
    scene = bpy.context.scene
    scene.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        scene.vibrt_spp = spp
    else:
        scene.cycles.samples = spp
        scene.cycles.use_adaptive_sampling = False
    scene.cycles.diffuse_bounces = db
    scene.render.resolution_x = 720
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = percentage
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "32"
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
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
    print("=== building fresh scene ===", flush=True)
    build_scene()
    out_v = HERE / "test_scenes" / "_cornell_texsky_vibrt.exr"
    out_c = HERE / "test_scenes" / "_cornell_texsky_cycles.exr"
    print("=== render vibrt ===", flush=True)
    render("vibrt", 256, 0, out_v, 50)
    print("=== render cycles ===", flush=True)
    render("cycles", 256, 0, out_c, 50)
    v = stats(out_v)
    c = stats(out_c)
    print(f"\n=== Cornell box-style TexSky test ===")
    print(f"  vibrt  mean RGB = ({v[..., 0].mean():.3f}, {v[..., 1].mean():.3f}, {v[..., 2].mean():.3f})")
    print(f"  cycles mean RGB = ({c[..., 0].mean():.3f}, {c[..., 1].mean():.3f}, {c[..., 2].mean():.3f})")
    print(f"  vibrt  peak  = {v.max():.2f}, p99 = {np.percentile(v, 99):.2f}")
    print(f"  cycles peak  = {c.max():.2f}, p99 = {np.percentile(c, 99):.2f}")
    print(f"  total ratio v/c = {v.sum() / max(c.sum(), 1e-6):.4f}")


main()
