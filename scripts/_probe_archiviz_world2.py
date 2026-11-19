"""Dump every link in the flat-archiviz world node tree to figure out how the
two Backgrounds and the Light Path node are wired into Mix Shader."""
import bpy

w = bpy.context.scene.world
nt = w.node_tree
print(f"=== world '{w.name}' nodes/links ===")
for n in nt.nodes:
    print(f"node: {n.name} ({n.bl_idname})")
print()
for link in nt.links:
    fr = link.from_node.name
    fs = link.from_socket.name
    tn = link.to_node.name
    ts = link.to_socket.name
    print(f"  {fr}.{fs}  ->  {tn}.{ts}")
