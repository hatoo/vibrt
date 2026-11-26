"""Run the exporter on pabellon and print the `world` section of the
JSON it produces, so we can confirm projection / extension are emitted
correctly for layer B (the FLAT-projection JPG)."""
import bpy, os, sys, json
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
from vibrt_blender.export_context import ExportContext
export_context = ExportContext()
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

from vibrt_blender.exporter import (
    export_scene_to_memory, prebake_sky_envmaps_for_world,
)
prebake_sky_envmaps_for_world(bpy.context.scene.world, w=2048, h=1024, context=export_context)
deps = bpy.context.evaluated_depsgraph_get()
json_str, _bin, _texs = export_scene_to_memory(deps, texture_pct=5, context=export_context)
data = json.loads(json_str)
world = data.get("world")
if world is None:
    print("world is None")
else:
    print("=== world JSON ===")
    print(json.dumps(world, indent=2))
print("\n=== lights JSON ===")
arr = data.get("lights", [])
print(f"  total: {len(arr)}")
for L in arr:
    print(f"  {L}")
