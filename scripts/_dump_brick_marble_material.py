"""Dump 'brick marble smooth' material's shader graph. Goal: find which
Principled inputs vibrt and cycles disagree on at sun-direction NEE eval.
"""
import bpy

mat = bpy.data.materials.get("brick marble smooth")
if mat is None:
    print("not found")
    raise SystemExit

print(f"=== material: '{mat.name}' ===")
nt = mat.node_tree
out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output), None)
print(f"  output = {out.name}")

def fmt(v):
    if hasattr(v, "__iter__"):
        return tuple(round(float(x), 4) for x in v)
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    return v


def walk(node, depth=0, seen=None):
    if seen is None:
        seen = set()
    if node.name in seen:
        print(f"{'  '*depth}↻ {node.bl_idname}/{node.name}")
        return
    seen.add(node.name)
    label = f"{node.bl_idname}/{node.name}"
    if hasattr(node, "image") and node.image:
        label += f" image={node.image.name!r} cs={node.image.colorspace_settings.name}"
    if node.bl_idname in {"ShaderNodeMath", "ShaderNodeVectorMath"}:
        label += f" op={node.operation}"
    if node.bl_idname == "ShaderNodeMixRGB":
        label += f" blend={node.blend_type}"
    print(f"{'  '*depth}{label}")
    for inp in node.inputs:
        if inp.is_linked:
            src = inp.links[0].from_node
            ssock = inp.links[0].from_socket
            print(f"{'  '*(depth+1)}in '{inp.name}' ← {src.bl_idname}/{src.name}.{ssock.name}")
            walk(src, depth + 2, seen)
        else:
            try:
                v = inp.default_value
                print(f"{'  '*(depth+1)}in '{inp.name}' = {fmt(v)}")
            except Exception:
                pass


surf = out.inputs.get("Surface")
if surf and surf.is_linked:
    src = surf.links[0].from_node
    walk(src)
