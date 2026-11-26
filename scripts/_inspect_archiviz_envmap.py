"""Run the addon's prebake + export on flat-archiviz, then dump the world
envmap pixel statistics so we can see whether the bake produced uniform
blackbody radiance or has black/missing regions."""
import bpy, json, sys, os, numpy as np

sys.path.insert(0, os.path.abspath("blender"))
import vibrt_blender.exporter as exporter
from vibrt_blender.export_context import ExportContext
export_context = ExportContext()

# Mirror engine.update(): pre-bake the world.
exporter.clear_sky_bake_cache(context=export_context)
exporter.prebake_sky_envmaps_for_world(bpy.context.scene.world, context=export_context)

depsgraph = bpy.context.evaluated_depsgraph_get()
scene_json_str, _, tex_arrays = exporter.export_scene_to_memory(depsgraph, context=export_context)
scene = json.loads(scene_json_str)

# Find the envmap texture id from scene.world.
print(json.dumps(scene.get("world", {}), indent=2))

w_desc = scene.get("world", {})
tex_id = None
if w_desc.get("type") == "envmap":
    tex_id = w_desc["texture"]
elif w_desc.get("type") == "mixed":
    print("(mixed world; checking layer a)")
    tex_id = w_desc["a"].get("texture")

if tex_id is None:
    print("no envmap texture id")
    sys.exit(0)

# Resolve TextureDesc for that id.
td = scene["textures"][tex_id]
print(f"texture[{tex_id}] desc: {td}")
ai = td["array_index"]
arr = np.asarray(tex_arrays[ai], dtype=np.float32)
print(f"texture[{tex_id}] pixels: {len(arr)} floats  ({td['width']}x{td['height']}x{td.get('channels','?')})")

# Reshape (height, width, channels)
ch = td.get("channels", 3)
arr = arr.reshape(td["height"], td["width"], ch)
print(f"min/max/mean per channel:")
for c, name in enumerate("rgb"[:ch]):
    sub = arr[..., c]
    print(f"  {name}: min={sub.min():.4f} max={sub.max():.4f} mean={sub.mean():.4f}")
# How many pixels are pure black (all 3 channels < 1e-4)?
mask = np.all(arr[..., :3] < 1e-4, axis=2)
print(f"black pixels (RGB<1e-4): {mask.sum()} / {td['height']*td['width']} = {mask.mean()*100:.2f}%")
# Sample a few directions
H, W = td["height"], td["width"]
for y, x in [(0, 0), (H//4, W//4), (H//2, W//2), (3*H//4, W//4), (H-1, W-1)]:
    print(f"  px({x:4d},{y:4d}) = {tuple(round(float(v), 4) for v in arr[y, x, :3])}")
