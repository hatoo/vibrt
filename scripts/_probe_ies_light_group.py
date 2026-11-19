"""Drill into the ShaderNodeGroup that wraps the IES on each light in
ies_light.blend, so we can see the exact node graph the exporter has
to recognise.
"""
import bpy

for obj in bpy.context.scene.objects:
    if obj.type != 'LIGHT':
        continue
    light = obj.data
    if not light.use_nodes or light.node_tree is None:
        continue
    for n in light.node_tree.nodes:
        if n.bl_idname == 'ShaderNodeGroup':
            tree = n.node_tree
            print(f"\n=== [{obj.name}] group {n.name!r} -> NodeTree {tree.name!r} ===")
            for inner in tree.nodes:
                kind = inner.bl_idname
                marker = ""
                if kind == "ShaderNodeTexIES":
                    marker = (
                        f" mode={inner.mode} filepath={inner.filepath!r} "
                        f"ies={(inner.ies.name if inner.ies else None)!r}"
                    )
                print(f"  {inner.name!r:30s} bl_idname={kind}{marker}")
            print("  links:")
            for lk in tree.links:
                print(f"    {lk.from_node.name}.{lk.from_socket.name} -> "
                      f"{lk.to_node.name}.{lk.to_socket.name}")
            # Show what the outer Group node feeds into.
            print("  outer links from this Group:")
            for lk in light.node_tree.links:
                if lk.from_node is n:
                    print(f"    Group.{lk.from_socket.name} -> "
                          f"{lk.to_node.name}.{lk.to_socket.name}")
                if lk.to_node is n:
                    print(f"    {lk.from_node.name}.{lk.from_socket.name} -> "
                          f"Group.{lk.to_socket.name}")
            # Group node default values for any inputs.
            for sock in n.inputs:
                if not sock.is_linked:
                    val = sock.default_value
                    if hasattr(val, "__len__"):
                        val = list(val)
                    print(f"  outer input {sock.name!r} = {val!r}")
