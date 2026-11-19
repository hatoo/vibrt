"""Inspect material/object structure for the pabellon water and pond.

We saw vibrt rendering the pond as solid white instead of the transparent
water Cycles produces. The likely culprits are: a Glass BSDF that vibrt
isn't reading, a transparency input the exporter mishandles, or an
object that's being treated as opaque.
"""
import bpy

scene = bpy.context.scene
print(f"scene: {scene.name}, world: {scene.world.name}")

# Hunt for "water"-named materials and dump their key Principled / BSDF inputs.
def shader_summary(node, depth=0):
    pad = "  " * depth
    extra = ""
    if hasattr(node, "name"):
        extra += f" '{node.name}'"
    print(f"{pad}{node.bl_idname}{extra}")

def walk_back(sock, depth=0, seen=None):
    if seen is None:
        seen = set()
    if not sock.is_linked:
        try:
            dv = sock.default_value
            if hasattr(dv, '__len__'):
                v = "(" + ", ".join(f"{x:.3f}" for x in dv) + ")"
            else:
                v = f"{dv:.4f}" if isinstance(dv, float) else str(dv)
            return v
        except Exception:
            return "?"
    n = sock.links[0].from_node
    if n.name in seen:
        return f"<cycle:{n.bl_idname}>"
    seen = seen | {n.name}
    return f"{n.bl_idname}.{sock.links[0].from_socket.name}"

for mat in bpy.data.materials:
    if "water" not in mat.name.lower() and "pond" not in mat.name.lower() and "lotus" not in mat.name.lower():
        continue
    print(f"\n=== material '{mat.name}' ===")
    if not mat.use_nodes or mat.node_tree is None:
        print("  (no nodes)")
        continue
    out = next((n for n in mat.node_tree.nodes
                if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output),
               None) or next((n for n in mat.node_tree.nodes
                              if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    if out is None:
        print("  (no output)")
        continue
    surf = out.inputs.get("Surface")
    print(f"  Surface ← {walk_back(surf)}")
    if surf and surf.is_linked:
        src = surf.links[0].from_node
        print(f"  driver: {src.bl_idname} '{src.name}'")
        for inp in src.inputs:
            print(f"    {inp.name:20s}: {walk_back(inp)}")

print("\n=== objects with 'water'/'pond' in name ===")
for obj in scene.objects:
    if "water" in obj.name.lower() or "pond" in obj.name.lower():
        mats = [s.material.name if s.material else "(none)" for s in obj.material_slots]
        print(f"  {obj.name:30s} type={obj.type} mats={mats}")
