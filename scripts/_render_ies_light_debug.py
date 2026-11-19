"""Debug: render ies_light through vibrt_native directly (bypassing
Blender's render pipeline) and print pixel statistics on the raw float
buffer. Lets us see if anything is being computed regardless of the
PNG/tonemap conversion.
"""
import bpy
import numpy as np
import sys

bpy.context.scene.render.engine = "VIBRT"
sys.path.insert(0, r"C:\Users\hato2\Desktop\vibrt\blender")
from vibrt_blender import exporter, runner

dg = bpy.context.evaluated_depsgraph_get()
json_str, mesh_blobs, texture_arrays = exporter.export_scene_to_memory(dg)
print(f"json {len(json_str)} chars, {len(mesh_blobs)} mesh blobs, "
      f"{len(texture_arrays)} texture arrays")

def report(msg, *args, **kw):
    pass

def cancel():
    return False

pixels, depth = runner.run_render_inproc(
    json_str, mesh_blobs, report, cancel,
    denoise=False, texture_arrays=texture_arrays,
)
print(f"pixels shape={pixels.shape} dtype={pixels.dtype}")
print(f"  RGB min={pixels[..., :3].min():.6f} max={pixels[..., :3].max():.6f} "
      f"mean={pixels[..., :3].mean():.6f}")
print(f"  R: min={pixels[..., 0].min():.6f} max={pixels[..., 0].max():.6f} "
      f"mean={pixels[..., 0].mean():.6f}")
print(f"  G: min={pixels[..., 1].min():.6f} max={pixels[..., 1].max():.6f} "
      f"mean={pixels[..., 1].mean():.6f}")
print(f"  B: min={pixels[..., 2].min():.6f} max={pixels[..., 2].max():.6f} "
      f"mean={pixels[..., 2].mean():.6f}")
print(f"  alpha: min={pixels[..., 3].min():.6f} max={pixels[..., 3].max():.6f}")
print(f"  nonzero RGB pixels: {(pixels[..., :3].sum(-1) > 0).sum()} / "
      f"{pixels.shape[0]*pixels.shape[1]}")
print(f"depth: min={depth.min():.6f} max={depth.max():.6f} mean={depth.mean():.6f}")
print(f"  hit pixels (depth>0): {(depth > 0).sum()} / {depth.shape[0]*depth.shape[1]}")
