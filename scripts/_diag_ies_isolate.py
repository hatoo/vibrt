"""Isolate ies_light per-light-type IES delivery: render each of the 3 lights
alone (point/spot/area) in both engines, compare mean. Pinpoints which light
type's IES calibration diverges from Cycles.

Run: blender -b ies_light.blend --python scripts/_diag_ies_isolate.py -- <idx|all>
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
sel = argv[argv.index("--") + 1] if "--" in argv else "all"

scene = bpy.context.scene
lights = [o for o in scene.objects if o.type == "LIGHT"]
lights.sort(key=lambda o: o.name)
print("lights:", [(o.name, o.data.type) for o in lights])

OUT = HERE / "test_scenes" / "_eng_diff"
OUT.mkdir(parents=True, exist_ok=True)
SPP, PCT = 128, 100


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


def render(engine, tag):
    configure(engine)
    p = OUT / f"ies_iso_{tag}_{engine}.exr"
    bpy.context.scene.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(p))
    img.colorspace_settings.name = "Linear Rec.709"
    ww, hh = int(img.size[0]), int(img.size[1])
    flat = np.empty(ww * hh * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    bpy.data.images.remove(img)
    return flat.reshape(hh, ww, 4)[..., :3].reshape(-1, 3).mean(axis=0)


def set_only(idx):
    for i, o in enumerate(lights):
        o.hide_render = (idx is not None and i != idx)


cases = range(len(lights)) if sel == "all" else [int(sel)]
for idx in cases:
    set_only(idx)
    name = lights[idx].name
    lt = lights[idx].data.type
    v = render("vibrt", f"{idx}")
    c = render("cycles", f"{idx}")
    r = v / np.maximum(c, 1e-9)
    print(f"[{idx}] {name} ({lt}): vibrt={np.round(v,4)} cycles={np.round(c,4)} ratio={np.round(r,3)}")
print("=== DONE ===", flush=True)
