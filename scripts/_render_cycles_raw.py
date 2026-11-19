"""Render lone_monk with Cycles but with the compositor disabled, so the
output is the raw Combined pass (matching what vibrt produces).
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = Path(argv[0]) if argv else Path("preview_cycles_raw.png")
spp = int(argv[1]) if len(argv) > 1 else 64

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = spp
scene.cycles.use_adaptive_sampling = False

# Disable the compositor entirely (Mix / Mist / Glare / LensDist).
scene.render.use_compositing = False
scene.render.use_sequencer = False

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

with tempfile.TemporaryDirectory(prefix="cycles_raw_") as td:
    scene.render.filepath = str(Path(td) / "frame")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    bpy.ops.render.render(write_still=True)
    produced = next(iter(Path(td).glob("*.png")), None)
    if produced is None:
        raise SystemExit("no PNG produced")
    shutil.move(str(produced), str(out))

print(f"wrote {out}")
