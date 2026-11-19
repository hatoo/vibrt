"""Dump the exported material params for 'Lamp Gubi' so we can see whether
its emissive Principled BSDF + Blackbody color survived the export."""
import bpy, sys, os, json
sys.path.insert(0, os.path.abspath("blender"))
import vibrt_blender.exporter as exporter

# Pre-bake (engine.update equivalent) so world-bake doesn't error in deps.
exporter.clear_sky_bake_cache()
try:
    exporter.prebake_sky_envmaps_for_world(bpy.context.scene.world)
except Exception:
    pass

depsgraph = bpy.context.evaluated_depsgraph_get()
scene_json_str, _, _ = exporter.export_scene_to_memory(depsgraph)
scene = json.loads(scene_json_str)

target_names = {"Lamp Gubi", "Lampshade"}
mats = scene.get("materials", [])
print(f"total materials: {len(mats)}")
print(f"first material keys: {list(mats[0].keys()) if mats else '(empty)'}")
for i, m in enumerate(mats):
    name = m.get("name") or m.get("_name") or m.get("debug_name") or "?"
    if name in target_names:
        print(f"\n=== material[{i}] '{name}' ===")
        for k, v in m.items():
            if k == "color_graph":
                print(f"  {k}: <{len(v)} ops>")
            else:
                print(f"  {k}: {v}")
# fallback: scan all materials, print emission-related fields
print()
print("=== materials with non-zero emission ===")
for i, m in enumerate(mats):
    em = m.get("emission") or m.get("emission_color")
    es = m.get("emission_strength")
    name = m.get("name") or m.get("_name") or "?"
    if em and any(c != 0 for c in (em if isinstance(em, list) else [em])):
        print(f"  [{i}] {name} emission={em} strength={es}")
