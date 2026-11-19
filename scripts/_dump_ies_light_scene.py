"""Run the in-process exporter on ies_light.blend and dump the JSON to
stdout (no render). For diagnosing why vibrt produces all-black on this
scene."""
import bpy
import json
import sys

# Force engine to VIBRT so the addon is alive.
bpy.context.scene.render.engine = "VIBRT"

# Ensure the addon is registered
import importlib
parent_dir = r"C:\Users\hato2\Desktop\vibrt\blender"
sys.path.insert(0, parent_dir)
exporter = importlib.import_module("vibrt_blender.exporter")

depsgraph = bpy.context.evaluated_depsgraph_get()
json_str, mesh_blobs, texture_arrays = exporter.export_scene_to_memory(depsgraph)
data = json.loads(json_str)

print("=== render ===")
print(json.dumps(data["render"], indent=2))
print("=== camera ===")
print(json.dumps(data["camera"], indent=2))
print("=== lights ===")
for L in data["lights"]:
    print(json.dumps(L, indent=2))
print("=== materials ===")
for m in data["materials"]:
    # Drop the colour-graph blobs to keep output readable
    m_short = {k: v for k, v in m.items() if k not in ("color_graph",)}
    print(json.dumps(m_short, indent=2))
print("=== meshes (truncated) ===")
for M in data["meshes"]:
    print(f"  {M.get('name', '?')}: vertices_blob={M.get('vertices', {}).get('len', '?')} bytes, "
          f"indices_blob={M.get('indices', {}).get('len', '?')} bytes, "
          f"material_indices_count={len(M.get('material_indices', []))}")
print("=== objects ===")
for o in data["objects"]:
    short = {k: v for k, v in o.items() if k not in ("transform",)}
    short["transform_first6"] = o.get("transform", [])[:6]
    print(json.dumps(short, indent=2))
print("=== world ===")
print(json.dumps(data.get("world"), indent=2))
