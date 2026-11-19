"""Test the zenith-sun-disc hypothesis for pabellon's B-over.

Renders fac=0 (sky-only) in both engines with sun_disc ON vs OFF. If the
B-over shrinks / engines converge when sun_disc is OFF, then vibrt mishandles
the zenith sun disc (equirect pole singularity kills its NEE contribution),
while Cycles samples it analytically.

Run: blender -b pabellon.blend --python scripts/_diag_pabellon_sundisc.py -- <on|off>
"""
import sys
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
mode = argv[argv.index("--") + 1] if "--" in argv else "on"
disc = (mode == "on")

scene = bpy.context.scene
world = scene.world
mix = next(n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeMixShader")
mix.inputs[0].default_value = 0.0  # sky only
sky = next(n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeTexSky")
sky.sun_disc = disc
print(f"sun_disc = {sky.sun_disc}  (elevation={sky.sun_elevation:.4f})")

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
    p = OUT / f"pab_disc_{mode}_{engine}.exr"
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
    return flat.reshape(hh, ww, 4)[..., :3].copy()


v = load(render("vibrt"))
c = load(render("cycles"))
vm = v.reshape(-1, 3).mean(axis=0)
cm = c.reshape(-1, 3).mean(axis=0)
print(f"\nsun_disc={mode}  full-frame vibrt={np.round(vm,4)} cycles={np.round(cm,4)} ratio={np.round(vm/np.maximum(cm,1e-9),3)}")
print("=== DONE ===", flush=True)
