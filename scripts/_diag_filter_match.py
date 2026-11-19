"""Test whether the Cycles pixel-filter (default Blackman-Harris 1.5px) vs
vibrt's box 1px is inflating the low-contrast pattern. Renders a scene with
Cycles forced to BOX width 1.0 (matching vibrt) and compares per-bucket ratio.

Run: blender -b <scene>.blend --python scripts/_diag_filter_match.py -- <tag> <box|default>
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
rest = argv[argv.index("--") + 1:] if "--" in argv else ["scene", "box"]
tag, mode = rest[0], rest[1]
OUT = HERE / "test_scenes" / "_eng_diff"
SPP, PCT = 64, 25


def cfg(engine):
    s = bpy.context.scene
    s.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        s.vibrt_spp = SPP
    else:
        s.cycles.samples = SPP
        s.cycles.use_adaptive_sampling = False
        if mode == "box":
            s.cycles.pixel_filter_type = 'BOX'
            s.cycles.filter_width = 1.0
    s.render.resolution_percentage = PCT
    s.render.image_settings.file_format = "OPEN_EXR"
    s.render.image_settings.color_depth = "32"
    s.view_settings.view_transform = "Raw"; s.view_settings.look = "None"
    s.view_settings.exposure = 0.0; s.view_settings.gamma = 1.0
    s.render.use_compositing = False


def render(engine):
    cfg(engine)
    p = OUT / f"fm_{tag}_{mode}_{engine}.exr"
    bpy.context.scene.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(p)); img.colorspace_settings.name = "Linear Rec.709"
    w, h = int(img.size[0]), int(img.size[1])
    fl = np.empty(w*h*4, np.float32); img.pixels.foreach_get(fl); bpy.data.images.remove(img)
    return fl.reshape(h, w, 4)[..., :3].reshape(-1, 3)


v = render("vibrt"); c = render("cycles")
clum = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
print(f"\n=== {tag} filter={mode} ===")
print(f"overall ratio = {np.round(v.mean(0)/np.maximum(c.mean(0),1e-9),3)}")
for lo, hi, t in [(0, 0.3, 'dim'), (0.3, 1.0, 'mid'), (1.0, 1e9, 'bright')]:
    m = (clum >= lo) & (clum < hi)
    if m.sum() == 0: continue
    print(f"  {t:7s} n={int(m.sum()):6d} ratio={np.round(v[m].mean(0)/np.maximum(c[m].mean(0),1e-9),3)}")
print("=== DONE ===", flush=True)
