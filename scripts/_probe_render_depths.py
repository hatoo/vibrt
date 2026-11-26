"""Render lone_monk in vibrt and dump the depth at the sampled regions."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))

import bpy
import vibrt_blender.exporter as exporter
from vibrt_blender.export_context import ExportContext
export_context = ExportContext()
import vibrt_blender.runner as runner

scene = bpy.context.scene
exporter.clear_sky_bake_cache(context=export_context)
exporter.prebake_sky_envmaps_for_world(scene.world, context=export_context)
depsgraph = bpy.context.evaluated_depsgraph_get()
json_str, mesh_blobs, texture_arrays = exporter.export_scene_to_memory(depsgraph, context=export_context)

def report(*a, **kw):
    pass

class _Br:
    def __call__(self): return False
    def __bool__(self): return False

pixels, depth = runner.run_render_inproc(
    json_str, mesh_blobs, lambda *a, **kw: None, _Br(),
    denoise=False, texture_arrays=texture_arrays,
)

# Render is 1440x1080. The PNG previews we sampled were 600x450 (downscaled
# to 600x450 it seems, but actually previews stay at full res).
print(f"depth shape: {depth.shape}")
h, w = depth.shape

# Sample regions in 1440x1080:
# directly-lit-mid-distance was (260, 180, 360, 220) on 600x450
# scale factor 1440/600 = 2.4, 1080/450 = 2.4
def scale(reg, sx, sy):
    return (int(reg[0]*sx), int(reg[1]*sy), int(reg[2]*sx), int(reg[3]*sy))

regions = {
    "directly-lit-mid-distance": scale((260, 180, 360, 220), w/600, h/450),
    "shadowed-foreground":       scale((100, 350, 200, 410), w/600, h/450),
    "archway-ceiling":           scale((170,  30, 430,  70), w/600, h/450),
}

# vibrt depth has top-left origin (consistent with PNG). PIL sampling above
# also reads top-left. So same coords.
import math
mist_start = scene.world.mist_settings.start
mist_depth_range = scene.world.mist_settings.depth
print(f"Mist settings: start={mist_start}, depth={mist_depth_range}")
for name, region in regions.items():
    x0, y0, x1, y1 = region
    crop = depth[y0:y1, x0:x1]
    valid = crop[crop < 49.9]  # exclude misses (clip_end=50)
    if valid.size == 0:
        valid = crop
    print(f"\n=== {name} {region} ===")
    print(f"  depth: min={crop.min():.2f} max={crop.max():.2f} mean={crop.mean():.2f} valid_mean={valid.mean():.2f}")
    mist = np.clip((valid - mist_start) / max(mist_depth_range, 1e-6), 0, 1)
    print(f"  mist:  min={mist.min():.4f} max={mist.max():.4f} mean={mist.mean():.4f}")
