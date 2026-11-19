"""Find every emissive material and dump its full graph."""
import bpy

def walk(n, depth=0, seen=None):
    if seen is None: seen = set()
    if n.name in seen: return
    seen = seen | {n.name}
    indent = "  " * depth
    print(f"{indent}{n.bl_idname} '{n.name}'")
    for inp in n.inputs:
        if inp.is_linked:
            link = inp.links[0]
            print(f"{indent}  in {inp.name}: {link.from_node.bl_idname}.{link.from_socket.name}")
            walk(link.from_node, depth + 2, seen)
        else:
            try:
                dv = inp.default_value
                if hasattr(dv, '__len__'):
                    val = "(" + ", ".join(f"{v:.3f}" for v in dv) + ")"
                else:
                    val = f"{dv:.4f}" if isinstance(dv, float) else str(dv)
                print(f"{indent}  in {inp.name} = {val}")
            except Exception:
                pass

# Look for materials with Emission node
for mat in bpy.data.materials:
    if mat.node_tree is None or not mat.use_nodes: continue
    has_emission = any(n.bl_idname in ("ShaderNodeEmission", "ShaderNodeBsdfPrincipled")
                       and (n.bl_idname == "ShaderNodeEmission"
                            or any(inp.is_linked or (inp.name in ("Emission", "Emission Color")
                                                     and any(c > 0.01 for c in inp.default_value[:3])
                                                     if hasattr(inp, "default_value") and hasattr(inp.default_value, "__len__")
                                                     else False)
                                   for inp in n.inputs if inp.name in ("Emission", "Emission Color")))
                       for n in mat.node_tree.nodes)
    name_l = mat.name.lower()
    if "candle" in name_l or "flame" in name_l or "emit" in name_l or "light" in name_l:
        print(f"\n========== material '{mat.name}' (by name) ==========")
        out = next((n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"
                    and n.is_active_output), None) or next(
            (n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
        if out:
            walk(out)
