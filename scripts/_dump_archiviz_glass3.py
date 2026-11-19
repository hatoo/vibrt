"""Look at the window glass NodeGroup contents and how the exporter mapped it."""
import bpy, json, os, sys

sys.path.insert(0, os.path.abspath("blender"))
import vibrt_blender.exporter as exporter  # noqa: E402


def describe_tree(node_tree, indent=""):
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
        print(f"{indent}(no surface link)")
        return
    descend(surf.links[0].from_node, 0, indent)


def descend(node, depth, indent_base, max_depth=6):
    ind = indent_base + "  " * (depth + 1)
    print(f"{ind}{node.bl_idname}: {node.name!r}")
    if depth >= max_depth:
        return
    if node.bl_idname == "ShaderNodeGroup" and getattr(node, "node_tree", None):
        print(f"{ind}  >> entering group {node.node_tree.name!r}")
        # Find the group output
        gout = None
        for gn in node.node_tree.nodes:
            if gn.bl_idname == "NodeGroupOutput":
                gout = gn
                break
        if gout:
            for inp in gout.inputs:
                if inp.is_linked:
                    print(f"{ind}    group output {inp.name!r} ← {inp.links[0].from_node.name!r}.{inp.links[0].from_socket.name}")
                    descend(inp.links[0].from_node, depth + 1, indent_base, max_depth)
    for inp in node.inputs:
        if inp.is_linked:
            from_node = inp.links[0].from_node
            from_sock = inp.links[0].from_socket.name
            print(f"{ind}  in {inp.name} ← {from_node.name!r}.{from_sock}")
            if depth + 1 < max_depth:
                descend(from_node, depth + 1, indent_base, max_depth)


# All four candidate window glass materials
for name in ("Glass Plane Library", "Glass Window", "Glass.001",
             "Glass Bottle", "Glass Kitchen", "glass Tv"):
    m = bpy.data.materials.get(name)
    if m:
        print(f"=== material tree for {name!r} ===")
        describe_tree(m.node_tree)
        print()

# Now run the exporter and dump the FULL exported JSON for each
print("=== exporter output (full) ===")
depsgraph = bpy.context.evaluated_depsgraph_get()
scene_json_str, _, _ = exporter.export_scene_to_memory(depsgraph)
scene = json.loads(scene_json_str)
mats_out = scene.get("materials", [])
target = ("Glass Plane Library", "Glass Window", "Glass.001",
          "Glass Bottle", "Glass Kitchen", "glass Tv")
for m in mats_out:
    if m.get("name") in target:
        print(f"\n--- {m.get('name')!r} ---")
        # print ALL keys, not filtered
        print(json.dumps(m, indent=2, default=str))
