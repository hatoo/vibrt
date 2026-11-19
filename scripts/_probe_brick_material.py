"""Dump key Principled BSDF inputs for the brick clay material."""
import bpy

scene = bpy.context.scene
target = "brick clay"
mat = bpy.data.materials.get(target)
if mat is None:
    mat = next((m for m in bpy.data.materials if "brick" in m.name.lower()), None)

print(f"material: {mat.name if mat else '(not found)'}")
if mat is None:
    raise SystemExit
print(f"use_nodes: {mat.use_nodes}")
nt = mat.node_tree
out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"
            and n.is_active_output), None) or next(
    (n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
if out is None:
    raise SystemExit
sock = out.inputs["Surface"]
src = sock.links[0].from_node if sock.is_linked else None
print(f"Output.Surface ← {src.bl_idname if src else '(unlinked)'}")

# Walk back through Mix shaders / colour math to find a Principled BSDF
visited = set()
def walk(n, depth=0):
    if n.name in visited or depth > 10: return []
    visited.add(n.name)
    if n.bl_idname == "ShaderNodeBsdfPrincipled":
        return [n]
    out = []
    for s in n.inputs:
        if s.is_linked:
            out += walk(s.links[0].from_node, depth+1)
    return out

principled_nodes = walk(src) if src else []
print(f"Principled BSDF instances: {len(principled_nodes)}")
for p in principled_nodes:
    print(f"\n--- {p.name} ---")
    for inp in p.inputs:
        if inp.is_linked:
            link = inp.links[0]
            extra = ""
            if hasattr(inp, "default_value"):
                try:
                    dv = inp.default_value
                    if hasattr(dv, '__len__'):
                        extra = f" default=({', '.join(f'{v:.3f}' for v in dv)})"
                    else:
                        extra = f" default={dv}"
                except Exception:
                    pass
            print(f"  {inp.name:30s} <- {link.from_node.bl_idname}.{link.from_socket.name}{extra}")
        else:
            try:
                dv = inp.default_value
                if hasattr(dv, '__len__'):
                    val = "(" + ", ".join(f"{v:.3f}" for v in dv) + ")"
                else:
                    val = f"{dv:.4f}" if isinstance(dv, float) else f"{dv}"
                print(f"  {inp.name:30s} = {val}")
            except Exception:
                print(f"  {inp.name:30s} = ?")
