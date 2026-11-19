"""Inspect the tree, lotus_leaf, grass materials in pabellon to see what
vibrt is rendering grey/brown when Cycles renders them green/orange."""
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

# List all materials, then dump tree/lotus/leaf/grass
for mat in bpy.data.materials:
    n = mat.name.lower()
    if not any(k in n for k in ("tree", "leaf", "lotus", "grass", "branch", "bark", "trunk", "foliage", "plant", "wood")):
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
    walk(out)

# What objects use these materials (for context)
print("\n\n========== object/material usage ==========")
target_mats = {m.name for m in bpy.data.materials
               if any(k in m.name.lower() for k in ("tree", "leaf", "lotus", "grass", "branch", "bark", "foliage"))}
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    used = [s.material.name for s in obj.material_slots if s.material and s.material.name in target_mats]
    if used:
        print(f"  {obj.name:40s}: {used}")
