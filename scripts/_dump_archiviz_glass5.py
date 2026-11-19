"""Dump first material structure to see actual JSON fields."""
import bpy, json, os, sys

sys.path.insert(0, os.path.abspath("blender"))
import vibrt_blender.exporter as exporter  # noqa: E402

depsgraph = bpy.context.evaluated_depsgraph_get()
scene_json_str, _, _ = exporter.export_scene_to_memory(depsgraph)
scene = json.loads(scene_json_str)
mats_out = scene.get("materials", [])
print(f"=== first material keys ===")
if mats_out:
    print(list(mats_out[0].keys()))
print()
print(f"=== first material ===")
if mats_out:
    print(json.dumps(mats_out[0], indent=2, default=str)[:2000])

# How are materials referenced from objects? Likely via index.
print()
print(f"=== top-level scene keys ===")
print(list(scene.keys()))
print()
# objects[i].materials should map to materials list
objs = scene.get("objects", [])
print(f"=== first object material refs ===")
if objs:
    print(json.dumps({k: v for k, v in objs[0].items() if 'mat' in k.lower() or k == 'name'}, indent=2))
