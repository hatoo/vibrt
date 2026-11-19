"""Dump pabellon's world node tree so we can see what HDRI / cloud
textures and mappings drive the surface."""
import bpy

scene = bpy.context.scene
world = scene.world
print(f"world: {world.name}")

def walk(n, depth=0, seen=None):
    if seen is None: seen = set()
    if n.name in seen: return
    seen = seen | {n.name}
    indent = "  " * depth
    extra = ""
    if n.bl_idname == "ShaderNodeTexEnvironment":
        if n.image:
            extra = f" image='{n.image.name}' size={tuple(n.image.size)}"
        else:
            extra = " image=None"
    if n.bl_idname == "ShaderNodeMapping":
        loc = tuple(round(x, 3) for x in n.inputs["Location"].default_value)
        rot = tuple(round(x, 3) for x in n.inputs["Rotation"].default_value)
        scl = tuple(round(x, 3) for x in n.inputs["Scale"].default_value)
        extra = f" loc={loc} rot={rot} scl={scl}"
    print(f"{indent}{n.bl_idname} '{n.name}'{extra}")
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

out = next((n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeOutputWorld"), None)
walk(out)
