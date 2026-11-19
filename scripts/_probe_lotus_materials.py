"""Dump the full shader graphs for white_lotus*, lotus*, water materials so
we can see exactly what's failing in vibrt's material exporter."""
import bpy

names = ("white_lotus", "white_lotus_leafs", "white_lotus_orange",
         "lotus", "leafs", "stem", "flower")

def walk_node(n, depth=0, seen=None):
    if seen is None: seen = set()
    if n.name in seen: return
    seen = seen | {n.name}
    indent = "  " * depth
    print(f"{indent}{n.bl_idname} '{n.name}'")
    for inp in n.inputs:
        if inp.is_linked:
            link = inp.links[0]
            print(f"{indent}  in {inp.name}: {link.from_node.bl_idname}.{link.from_socket.name}")
            walk_node(link.from_node, depth + 2, seen)
        else:
            try:
                dv = inp.default_value
                if hasattr(dv, '__len__'):
                    val = "(" + ", ".join(f"{v:.3f}" for v in dv) + ")"
                else:
                    val = f"{dv:.4f}" if isinstance(dv, float) else str(dv)
                print(f"{indent}  in {inp.name} = {val}")
            except Exception:
                print(f"{indent}  in {inp.name} = ?")

for mat in bpy.data.materials:
    if not any(k in mat.name.lower() for k in ("lotus", "water")):
        continue
    print(f"\n========== material '{mat.name}' ==========")
    if not mat.use_nodes or mat.node_tree is None:
        continue
    out = next((n for n in mat.node_tree.nodes
                if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output),
               None) or next((n for n in mat.node_tree.nodes
                              if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    if out is None:
        continue
    walk_node(out)
