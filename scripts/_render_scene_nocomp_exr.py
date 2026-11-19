"""Per-scene EXR-based |diff| measurement with compositor=OFF, view=Raw.
Run inside Blender via:
  blender -b <scene>.blend --python scripts/_render_scene_nocomp_exr.py -- --tag <name>

Writes:
  test_scenes/_eng_diff/<tag>_vibrt.exr
  test_scenes/_eng_diff/<tag>_cycles.exr
And prints a single line:
  RESULT tag=<name> spp=<S> pct=<P> H=<h> W=<w> v_mean=R,G,B c_mean=R,G,B abs_diff=overall ratio=v/c v_p99=N c_p99=N v_peak=N c_peak=N
"""
from __future__ import annotations
import argparse
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


def configure(engine: str, spp: int, pct: int):
    s = bpy.context.scene
    s.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        s.vibrt_spp = spp
        _c = os.environ.get("VIBRT_CLAMP")
        if _c is not None:
            s.vibrt_clamp_indirect = float(_c)
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


def render_to(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(p)
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
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--spp", type=int, default=int(os.environ.get("SPP", "64")))
    ap.add_argument("--pct", type=int, default=int(os.environ.get("PCT", "25")))
    ap.add_argument("--scene", default=None)
    args = ap.parse_args(argv)

    if args.scene is not None:
        if args.scene in bpy.data.scenes:
            bpy.context.window.scene = bpy.data.scenes[args.scene]

    out_dir = HERE / "test_scenes" / "_eng_diff"
    out_v = out_dir / f"{args.tag}_vibrt.exr"
    out_c = out_dir / f"{args.tag}_cycles.exr"

    print(f"=== {args.tag} vibrt spp={args.spp} pct={args.pct} ===", flush=True)
    configure("vibrt", args.spp, args.pct)
    render_to(out_v)
    print(f"=== {args.tag} cycles spp={args.spp} pct={args.pct} ===", flush=True)
    configure("cycles", args.spp, args.pct)
    render_to(out_c)

    v = load_exr(out_v); c = load_exr(out_c)
    H = min(v.shape[0], c.shape[0]); W = min(v.shape[1], c.shape[1])
    v, c = v[:H, :W], c[:H, :W]

    abs_diff = np.abs(v - c)
    overall_abs = float(abs_diff.mean())
    v_mean = v.reshape(-1, 3).mean(axis=0)
    c_mean = c.reshape(-1, 3).mean(axis=0)
    v_p99 = float(np.percentile(v, 99))
    c_p99 = float(np.percentile(c, 99))
    v_peak = float(v.max()); c_peak = float(c.max())
    ratio = float(v.sum() / max(c.sum(), 1e-6))

    print(f"\nRESULT tag={args.tag} spp={args.spp} pct={args.pct} H={H} W={W} "
          f"v_mean=({v_mean[0]:.4f},{v_mean[1]:.4f},{v_mean[2]:.4f}) "
          f"c_mean=({c_mean[0]:.4f},{c_mean[1]:.4f},{c_mean[2]:.4f}) "
          f"abs_diff={overall_abs:.5f} ratio={ratio:.4f} "
          f"v_p99={v_p99:.3f} c_p99={c_p99:.3f} v_peak={v_peak:.2f} c_peak={c_peak:.2f}", flush=True)


main()
