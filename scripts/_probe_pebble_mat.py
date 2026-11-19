"""What material do pebble instances use?"""
import bpy

col = bpy.data.collections.get("pebbles")
if col:
    for obj in col.objects[:5]:
        mats = [s.material.name if s.material else "(none)" for s in obj.material_slots]
        print(f"  {obj.name}: mats={mats}")

# Now dump the material
for name in ("pebble", "pebbles", "rock", "stone"):
    mat = bpy.data.materials.get(name)
    if mat:
        print(f"\n=== material '{mat.name}' ===")
        if not mat.use_nodes:
            print("  no nodes")
            continue
        out = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
        def walk(n, depth=0, seen=None):
            if seen is None: seen = set()
            if n.name in seen: return
            seen = seen | {n.name}
            indent = "  " * depth
            print(f"{indent}{n.bl_idname} '{n.name}'")
            for inp in n.inputs:
                if inp.is_linked:
                    print(f"{indent}  in {inp.name}: {inp.links[0].from_node.bl_idname}.{inp.links[0].from_socket.name}")
                    walk(inp.links[0].from_node, depth + 2, seen)
                else:
                    try:
                        dv = inp.default_value
                        if hasattr(dv, '__len__'):
                            v = "(" + ", ".join(f"{x:.3f}" for x in dv) + ")"
                        else:
                            v = f"{dv:.4f}" if isinstance(dv, float) else str(dv)
                        print(f"{indent}  in {inp.name} = {v}")
                    except: pass
        walk(out)
