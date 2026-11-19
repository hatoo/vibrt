"""Inspect dayLight_portal material in classroom — does our exporter
detect this as a portal (alpha_blend=0 + skip BLAS for camera-hidden)?
"""
import bpy

mat = bpy.data.materials.get("dayLight_portal")
if mat is None or not mat.node_tree:
    print("dayLight_portal not found / no node_tree")
    raise SystemExit
out = next((n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
if out is None:
    print("no output node")
    raise SystemExit

def walk(node, depth=0, prefix=""):
    print("  " * depth + prefix + f"{node.bl_idname}/{node.name}")
    for inp in node.inputs:
        if inp.is_linked:
            for link in inp.links:
                walk(link.from_node, depth + 1, f"{inp.name} ← ")
        else:
            try:
                v = inp.default_value
                if hasattr(v, "__iter__"):
                    v = tuple(round(float(x), 3) for x in v)
                else:
                    v = round(float(v), 3) if isinstance(v, (int, float)) else v
                print("  " * (depth + 1) + f"{inp.name} = {v}")
            except Exception:
                pass

print(f"=== dayLight_portal ===")
walk(out)

# Also check whitePaintedWindow and frostedGlass
for name in ("whitePaintedWindow", "frostedGlass"):
    mat = bpy.data.materials.get(name)
    if mat is None or not mat.node_tree:
        continue
    print(f"\n=== {name} ===")
    out = next((n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is not None:
        walk(out)
