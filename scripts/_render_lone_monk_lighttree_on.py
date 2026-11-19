"""Full lone_monk with cycles use_light_tree=True (forced) and compositor=OFF.
Tests whether vibrt's 60% over-delivery vs cycles is caused by mesh-light NEE
asymmetry: vibrt has mesh-light NEE on, cycles' default (light_tree=False)
means mesh-light NEE is less efficient.
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


def configure(engine: str, spp: int, percentage: int, db: int, light_tree: bool):
    scene = bpy.context.scene
    scene.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        scene.vibrt_spp = spp
    else:
        scene.cycles.samples = spp
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_light_tree = light_tree
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

    out_v = HERE / "test_scenes" / "lone_monk" / "_lt_db0_vibrt.exr"
    out_c0 = HERE / "test_scenes" / "lone_monk" / "_lt_db0_cycles_ltOFF.exr"
    out_c1 = HERE / "test_scenes" / "lone_monk" / "_lt_db0_cycles_ltON.exr"

    print(f"=== vibrt db=0 ===", flush=True)
    configure("vibrt", spp, pct, 0, light_tree=False)
    render_to(out_v)
    print(f"=== cycles db=0 light_tree=OFF ===", flush=True)
    configure("cycles", spp, pct, 0, light_tree=False)
    render_to(out_c0)
    print(f"=== cycles db=0 light_tree=ON ===", flush=True)
    configure("cycles", spp, pct, 0, light_tree=True)
    render_to(out_c1)

    v = load_exr(out_v)
    c0 = load_exr(out_c0)
    c1 = load_exr(out_c1)

    # crop to common shape
    H = min(v.shape[0], c0.shape[0], c1.shape[0])
    W = min(v.shape[1], c0.shape[1], c1.shape[1])
    v, c0, c1 = v[:H, :W], c0[:H, :W], c1[:H, :W]

    print(f"\n=== db=0 light_tree comparison (compositor=OFF) ===")
    print(f"  vibrt              mean = ({v[..., 0].mean():.3f}, {v[..., 1].mean():.3f}, {v[..., 2].mean():.3f})  peak={v.max():.2f}")
    print(f"  cycles lt=OFF      mean = ({c0[..., 0].mean():.3f}, {c0[..., 1].mean():.3f}, {c0[..., 2].mean():.3f})  peak={c0.max():.2f}")
    print(f"  cycles lt=ON       mean = ({c1[..., 0].mean():.3f}, {c1[..., 1].mean():.3f}, {c1[..., 2].mean():.3f})  peak={c1.max():.2f}")
    print(f"  ratio v / c(lt=OFF) = {v.sum() / max(c0.sum(), 1e-6):.3f}")
    print(f"  ratio v / c(lt=ON)  = {v.sum() / max(c1.sum(), 1e-6):.3f}")
    print(f"  cycles lt(ON/OFF)   = {c1.sum() / max(c0.sum(), 1e-6):.3f}")


main()
