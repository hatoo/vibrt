"""Print full exported JSON for glass-like materials in flat_archiviz."""
import bpy, json, os, sys

sys.path.insert(0, os.path.abspath("blender"))
import vibrt_blender.exporter as exporter  # noqa: E402

# First find the window object — it's the largest mesh in the back.
print("=== large flat objects (window candidates) ===")
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    d = obj.dimensions
    # Window-like: large in 2 dims, thin in 1
    dims = sorted([d.x, d.y, d.z])
    if dims[2] > 1.0 and dims[1] > 1.0 and dims[0] < 0.2:
        slots = [s.material.name if s.material else "(none)" for s in obj.material_slots]
        print(f"  {obj.name!r}: dim=({d.x:.2f},{d.y:.2f},{d.z:.2f}) loc=({obj.location.x:.2f},{obj.location.y:.2f},{obj.location.z:.2f}) mats={slots}")

# Now also list materials whose name contains "window" or that look like glass
print("\n=== materials containing 'window'/'Window'/'glass' in name ===")
target_names = []
for mat in bpy.data.materials:
    if any(s in mat.name.lower() for s in ("window", "glass", "vetro", "cristall")):
        target_names.append(mat.name)
        print(f"  {mat.name!r}")

# Walk material output to find Mix Shader patterns
def describe_tree(node_tree, prefix=""):
    if node_tree is None:
        return
    out = None
    for n in node_tree.nodes:
        if n.bl_idname == "ShaderNodeOutputMaterial":
            out = n
            break
    if out is None:
        return
    surf = out.inputs.get("Surface")
    if surf is None or not surf.is_linked:
        print(f"{prefix}(no surface)")
        return
    def descend(node, depth):
        ind = "  " * (depth + 1)
        print(f"{ind}{node.bl_idname}: {node.name!r}")
        for inp in node.inputs:
            if inp.is_linked:
                from_node = inp.links[0].from_node
                from_sock = inp.links[0].from_socket.name
                print(f"{ind}  in {inp.name} ← {from_node.name!r}.{from_sock}")
                if depth < 4:
                    descend(from_node, depth + 1)
            else:
                try:
                    val = inp.default_value
                    if hasattr(val, "__iter__"):
                        val = list(val)
                    print(f"{ind}  in {inp.name} = {val}")
                except Exception:
                    pass
    descend(surf.links[0].from_node, 0)

print("\n=== material tree for 'Glass Plane Library' ===")
m = bpy.data.materials.get("Glass Plane Library")
if m:
    describe_tree(m.node_tree)

print("\n=== exported JSON for all glass-named materials ===")
depsgraph = bpy.context.evaluated_depsgraph_get()
scene_json_str, _, _ = exporter.export_scene_to_memory(depsgraph)
scene = json.loads(scene_json_str)
mats_out = scene.get("materials", [])
for m in mats_out:
    if m.get("name") in target_names:
        print(f"\n--- {m.get('name')!r} ---")
        for k in ("transmission", "ior", "roughness", "metallic",
                  "base_color", "alpha_threshold", "specular_ior_level",
                  "pure_glossy", "pure_diffuse", "translucent_weight",
                  "coat_weight", "transmission_tex"):
            if k in m:
                print(f"  {k}: {m[k]}")
