"""Render flat-archiviz with Cycles at a configurable clip_start, sample pixels.
We compare the same diagnostic pixels at clip_start=2.0 (default) vs 0.001 to
figure out whether Cycles' clip_start really clips primary-ray geometry."""
import bpy, os, sys, math
from pathlib import Path

scene = bpy.context.scene
cam = scene.camera
cam.data.clip_start = float(os.environ.get("CLIP", "0.001"))
out = Path(os.environ.get("OUTPNG", "test_scenes/flat_archiviz/_cycles_clip0.png"))

scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
for backend in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = backend
        break
    except TypeError:
        continue
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type in ("OPTIX", "CUDA")
scene.cycles.device = "GPU"
scene.cycles.samples = int(os.environ.get("SPP", "64"))
scene.cycles.use_denoising = True
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = str(out.absolute())
print(f"Cycles clip_start={cam.data.clip_start}, samples={scene.cycles.samples}, → {out}")
bpy.ops.render.render(write_still=True)

# Sample pixels
import numpy as np
from PIL import Image
im = np.array(Image.open(out))[:, :, :3]
print("=== Cycles diagnostic pixels ===")
H, W = im.shape[:2]
for label, (y, x) in [("TL_corner", (0, 0)), ("TL_inner", (60, 60)),
                      ("TL_arc", (0, 150)), ("ML_edge", (H//2, 0)),
                      ("BL_corner", (H-1, 0))]:
    print(f"  {label:12s} y={y:4d} x={x:4d}  rgb={tuple(im[y,x])}")
