"""Walk the full church brick material to see what HueSat / RGBCurve /
Mix is doing to the dim Non-Color values."""
import bpy

mat = bpy.data.materials["church brick"]
out = next((n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL"), None)


def walk(node, depth=0, prefix=""):
    print("  " * depth + prefix + f"{node.bl_idname}/{node.name}")
    # Print specific node properties
    if node.bl_idname == "ShaderNodeHueSaturation":
        # All inputs are sockets, just dump
        pass
    elif node.bl_idname == "ShaderNodeRGBCurve":
        for i, c in enumerate(node.mapping.curves):
            pts = [(round(p.location.x, 3), round(p.location.y, 3))
                   for p in c.points]
            ch = ['C', 'R', 'G', 'B'][i] if i < 4 else str(i)
            print("  " * (depth + 1) + f"  curve[{ch}]: {pts}")
    elif node.bl_idname == "ShaderNodeMix":
        bt = getattr(node, "blend_type", "MIX")
        dt = getattr(node, "data_type", "?")
        cf = getattr(node, "clamp_factor", "?")
        cr = getattr(node, "clamp_result", "?")
        print("  " * (depth + 1) + f"  blend={bt} data={dt} clamp_factor={cf} clamp_result={cr}")
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


walk(out)
