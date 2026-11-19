"""Find the index of glass-named materials and dump their parameters."""
import bpy, json, os, sys

sys.path.insert(0, os.path.abspath("blender"))
import vibrt_blender.exporter as exporter  # noqa: E402

# Patch the exporter so we record name → index mapping.
orig = exporter.export_scene_to_memory
name_index = {}

# We need to monkey-patch the materials list to record names.
# Easier: just iterate bpy.data.materials in the same order the exporter would.
# Actually the exporter assigns ids on first encounter — so we need to inspect
# its internal `material_index`. Let's just compare slot order in objects.

depsgraph = bpy.context.evaluated_depsgraph_get()
scene_json_str, _, _ = exporter.export_scene_to_memory(depsgraph)
scene = json.loads(scene_json_str)
mats_out = scene["materials"]
objs = scene["objects"]
meshes = scene["meshes"]
print(f"=== {len(mats_out)} materials, {len(objs)} objects, {len(meshes)} meshes ===")

# Build name→index by walking objects: each object has .materials[] (a list of
# indices into the global mat list), in slot order matching its bpy material_slots.
# But the exported json has only "material": int per object (single mat) or per-mesh
# material_indices.
# Look at object structure
print("=== first object keys ===")
print(list(objs[0].keys()))
print(json.dumps(objs[0], indent=2, default=str)[:500])

# We need to find the right object — let's find one of the window objects.
target_obj_names = ("Glass Window.002", "Window.085", "Window.157", "Window.124")
print("\n=== window-like objects ===")
for obj in objs:
    n = obj.get("name", "")
    if n in target_obj_names:
        print(f"  {n!r}: {obj}")

# If name is missing from objects, build via Blender side.
# Map blender object name → index by scanning bpy.data.objects in dependency order.
print("\n=== Blender side: window glass material indices ===")
# Walk depsgraph instances in order matching exporter
target_mat_names = ("Glass Window", "Glass Plane Library", "Glass.001",
                    "Glass Bottle", "Glass Kitchen", "glass Tv")
# The exporter assigns mat indices when first seeing each via objects' material_slots.
# Let's build the same by iterating depsgraph in same way.
mat_first_seen = {}
ord_idx = 0
for obj_inst in depsgraph.object_instances:
    obj = obj_inst.object
    if obj is None:
        continue
    if obj.type != "MESH":
        continue
    for slot in obj.material_slots:
        if slot.material and slot.material.name not in mat_first_seen:
            mat_first_seen[slot.material.name] = ord_idx
            ord_idx += 1
for tname in target_mat_names:
    if tname in mat_first_seen:
        idx = mat_first_seen[tname]
        if idx < len(mats_out):
            m = mats_out[idx]
            interesting = {k: m[k] for k in m
                           if k in ("transmission", "ior", "metallic",
                                    "base_color", "alpha_threshold",
                                    "pure_glossy", "pure_diffuse",
                                    "translucent_weight", "transmission_tex",
                                    "color_graph_nodes")}
            # truncate color_graph_nodes for readability
            if "color_graph_nodes" in interesting:
                cg = interesting.pop("color_graph_nodes")
                interesting["color_graph_node_count"] = len(cg)
            print(f"  {tname!r} idx={idx}: {interesting}")
        else:
            print(f"  {tname!r} idx={idx} OUT OF RANGE ({len(mats_out)} mats)")
    else:
        print(f"  {tname!r}: NOT IN BLENDER → mat_first_seen MISS")
