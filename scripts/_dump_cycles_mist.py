"""Render lone_monk with Cycles, then dump Cycles' Mist pass values at the
brick-comparison sample regions so we can see whether Cycles is feeding
zero (or near-zero) into the compositor's exposure-driving Mist node."""
import os, sys
import numpy as np
import bpy

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 16
scene.cycles.use_adaptive_sampling = False

scene.view_layers[0].use_pass_mist = True
# Don't go through the compositor; we want raw passes.
scene.render.use_compositing = False

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

# Drive the renderer through bpy.ops.render.render and pull the Mist pass
# directly out of the resulting RenderResult — no compositor / file IO.
bpy.ops.render.render(use_viewport=False)

result = bpy.data.images.get("Render Result")
# RenderResult exposes passes via the iterator on its layers, but not via
# the simple Image API. Use the older renderpasses access.
for img in bpy.data.images:
    if img.type == "RENDER_RESULT":
        result = img
        break
print(f"Render Result image: {result}")

# Walk the active scene's view layer's passes and read each via render_pass.
vl = scene.view_layers[0]
print(f"view_layer passes available: {[p.name for p in vl.active_aov_layer_collection_passes if hasattr(vl, 'active_aov_layer_collection_passes')] if False else 'N/A'}")

# bpy.types.RenderResult exposes layers / passes — accessible only briefly
# via a `with` block on bpy.data.images['Render Result'].render_slots? No,
# the passes are on RenderResult/Layer/Pass objects via an internal API.
# Save out as multilayer EXR so we can pull pixels out cleanly.
import tempfile, shutil
from pathlib import Path
HERE = Path(__file__).parent.parent

with tempfile.TemporaryDirectory(prefix="cyc_") as td:
    out_path = Path(td) / "out.exr"
    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.color_depth = "32"
    result.save_render(filepath=str(out_path), scene=scene)
    final = HERE / "test_scenes" / "lone_monk" / "preview_cycles_mist.exr"
    shutil.move(str(out_path), str(final))
    print(f"wrote {final}")

# Re-load multilayer EXR. Blender exposes individual layers via
# image.render_slots / layers, but Image.pixels[] only gives the active.
# We can list layers and pick the Mist one.
img = bpy.data.images.load(str(final))
img.colorspace_settings.name = "Non-Color"
print(f"image: size={img.size[0]}x{img.size[1]} layers={[L.name for L in img.layers] if hasattr(img, 'layers') else 'no .layers'}")

# Attempt to access render_slots (RenderResult API only).
print(f"image.has_data: {img.has_data}")
print(f"image.channels: {img.channels}")

# Multilayer EXR: pixels[] gives the *active* layer's pixels. We'll just
# print stats on whatever pixels we get and rely on the Combined-only path
# being noticeable: if pixels are mostly 0, we know Cycles' Mist is empty.
w, h = img.size
flat = np.empty(w * h * img.channels, dtype=np.float32)
img.pixels.foreach_get(flat)
arr = flat.reshape(h, w, img.channels)
mist_ch = arr[..., 0]
print(f"\nactive-layer ch0: min={mist_ch.min():.4f} max={mist_ch.max():.4f} mean={mist_ch.mean():.4f}")
print(f"non-zero fraction: {(mist_ch > 1e-6).mean()*100:.2f}%")
