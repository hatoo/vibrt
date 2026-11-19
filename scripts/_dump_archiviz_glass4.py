"""List ALL exported material names and find glass-named ones."""
import bpy, json, os, sys

sys.path.insert(0, os.path.abspath("blender"))
import vibrt_blender.exporter as exporter  # noqa: E402

depsgraph = bpy.context.evaluated_depsgraph_get()
scene_json_str, _, _ = exporter.export_scene_to_memory(depsgraph)
scene = json.loads(scene_json_str)
mats_out = scene.get("materials", [])
print(f"=== {len(mats_out)} materials in export ===")
for m in mats_out:
    n = m.get("name", "?")
    if any(s in n.lower() for s in ("glass", "window", "vetro", "cristall", "trasp", "trans")):
        keys_present = [k for k in ("transmission", "ior", "metallic", "base_color",
                                    "alpha_threshold", "pure_glossy", "translucent_weight")
                        if k in m]
        vals = {k: m[k] for k in keys_present}
        print(f"  {n!r}: {vals}")

print()
print("=== ALL exported names ===")
for m in mats_out:
    print(f"  {m.get('name','?')!r}")
