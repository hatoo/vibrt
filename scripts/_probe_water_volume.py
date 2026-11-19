"""Dump the water material's Volume input fully so we can see what's
inside the volume slot."""
import bpy

mat = bpy.data.materials.get("water")
print(f"=== material '{mat.name}' ===")

def walk(n, depth=0, seen=None):
    if seen is None: seen = set()
    if n.name in seen:
        return
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

out = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
print("Surface:")
if out.inputs["Surface"].is_linked:
    walk(out.inputs["Surface"].links[0].from_node)
print("Volume:")
if out.inputs["Volume"].is_linked:
    walk(out.inputs["Volume"].links[0].from_node)
else:
    print("  (unlinked)")

# Where is water material used?
print("\nObjects using 'water':")
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    for slot in obj.material_slots:
        if slot.material and slot.material.name == "water":
            print(f"  {obj.name}: location={tuple(round(x,3) for x in obj.location)} dim={tuple(round(d,3) for d in obj.dimensions)}")
