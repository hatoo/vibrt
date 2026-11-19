"""Run the exporter on ies_light and dump exactly what colors each light got."""
import bpy, json, os, sys
sys.path.insert(0, os.path.abspath("blender"))
import vibrt_blender.exporter as exporter  # noqa: E402

depsgraph = bpy.context.evaluated_depsgraph_get()
scene_json_str, _, _ = exporter.export_scene_to_memory(depsgraph)
scene = json.loads(scene_json_str)

print(f"=== exported lights ({len(scene['lights'])}) ===")
for L in scene["lights"]:
    summary = {k: v for k, v in L.items() if k != "ies"
               and k != "transform" and k != "light_rotation"}
    if "ies" in L:
        ies_summary = {k: L["ies"][k] for k in
                       ("peak_candela", "integral_norm",
                        "peak_absolute_candela")
                       if k in L["ies"]}
        summary["ies_summary"] = ies_summary
    print(json.dumps(summary, indent=2, default=str))
