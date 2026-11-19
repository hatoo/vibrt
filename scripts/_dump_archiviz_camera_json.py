"""Run the addon's exporter on flat-archiviz and print the camera block from the
emitted JSON, plus the resolved scene aspect ratio. We want to see exactly what
gets shipped to the device."""
import bpy, json, sys, os

# Stage the working-tree addon (same trick as scripts/_blender_render.py).
sys.path.insert(0, os.path.abspath("blender"))
import vibrt_blender.exporter as exporter  # noqa: E402

depsgraph = bpy.context.evaluated_depsgraph_get()
scene_json_str, _, _ = exporter.export_scene_to_memory(depsgraph)
scene = json.loads(scene_json_str)

print("=== camera block ===")
print(json.dumps(scene["camera"], indent=2))

rs = bpy.context.scene.render
print()
print(f"render.resolution = {rs.resolution_x} x {rs.resolution_y}  (pct={rs.resolution_percentage})")
print(f"image aspect      = {rs.resolution_x / rs.resolution_y:.6f}")
