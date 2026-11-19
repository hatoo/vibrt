"""Isolate pabellon's world B-over: render both engines with the world
MixShader fac forced to 0 (sky-only) and 1 (photo-only), compare the sky
region. Pinpoints whether layer A (sky bake) or layer B (flat photo) diverges.

Run: blender -b pabellon.blend --python scripts/_diag_pabellon_isolate.py -- <fac>
"""
import sys, os
from pathlib import Path
import numpy as np
import bpy

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

argv = sys.argv
fac = float(argv[argv.index("--") + 1]) if "--" in argv else 0.0

# Force the world MixShader fac.
w = bpy.context.scene.world
mix = next(n for n in w.node_tree.nodes if n.bl_idname == "ShaderNodeMixShader")
mix.inputs[0].default_value = fac
print(f"forced MixShader fac = {fac}")

OUT = HERE / "test_scenes" / "_eng_diff"
OUT.mkdir(parents=True, exist_ok=True)
SPP, PCT = 32, 25


def configure(engine):
    s = bpy.context.scene
    s.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        s.vibrt_spp = SPP
    else:
        s.cycles.samples = SPP
        s.cycles.use_adaptive_sampling = False
    s.render.resolution_percentage = PCT
    s.render.image_settings.file_format = "OPEN_EXR"
    s.render.image_settings.color_mode = "RGB"
    s.render.image_settings.color_depth = "32"
    s.view_settings.view_transform = "Raw"
    s.view_settings.look = "None"
    s.view_settings.exposure = 0.0
    s.view_settings.gamma = 1.0
    s.render.use_compositing = False


def render(engine):
    configure(engine)
    p = OUT / f"pab_iso_fac{fac}_{engine}.exr"
    bpy.context.scene.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    return p


def load(p):
    img = bpy.data.images.load(str(p))
    img.colorspace_settings.name = "Linear Rec.709"
    ww, hh = int(img.size[0]), int(img.size[1])
    flat = np.empty(ww * hh * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    bpy.data.images.remove(img)
    return flat.reshape(hh, ww, 4)[..., :3][::-1].copy()


pv = render("vibrt")
pc = render("cycles")
v = load(pv); c = load(pc)
h = v.shape[0]
top = int(h * 0.15)
vm = v[:top].reshape(-1, 3).mean(axis=0)
cm = c[:top].reshape(-1, 3).mean(axis=0)
fm_v = v.reshape(-1, 3).mean(axis=0)
fm_c = c.reshape(-1, 3).mean(axis=0)
print(f"\nfac={fac}  ({'sky-only' if fac==0 else 'photo-only' if fac==1 else 'mix'})")
print(f"  sky-region  vibrt={np.round(vm,4)} cycles={np.round(cm,4)} ratio={np.round(vm/np.maximum(cm,1e-9),3)}")
print(f"  full-frame  vibrt={np.round(fm_v,4)} cycles={np.round(fm_c,4)} ratio={np.round(fm_v/np.maximum(fm_c,1e-9),3)}")
print("=== DONE ===", flush=True)
