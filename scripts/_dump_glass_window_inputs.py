"""Dump every Principled BSDF input for the Glass Window material."""
import bpy

m = bpy.data.materials.get("Glass Window")
if m is None:
    print("Glass Window not found")
    raise SystemExit
print(f"=== material '{m.name}' ===")
print(f"  blend_method:    {getattr(m, 'blend_method', '?')}")
print(f"  shadow_method:   {getattr(m, 'shadow_method', '?')}")
print(f"  alpha_threshold: {getattr(m, 'alpha_threshold', '?')}")
print(f"  use_backface_culling: {getattr(m, 'use_backface_culling', '?')}")
print(f"  diffuse_color:   {list(m.diffuse_color)}")
print(f"  surface_render_method: {getattr(m, 'surface_render_method', '?')}")

if m.node_tree:
    for n in m.node_tree.nodes:
        if n.bl_idname == "ShaderNodeBsdfPrincipled":
            print(f"\n=== Principled BSDF '{n.name}' inputs ===")
            for inp in n.inputs:
                if inp.is_linked:
                    src = inp.links[0].from_node.name
                    print(f"  {inp.name:30s} = LINKED from {src!r}.{inp.links[0].from_socket.name}")
                else:
                    try:
                        v = inp.default_value
                        if hasattr(v, "__iter__"):
                            v = list(v)
                        print(f"  {inp.name:30s} = {v}")
                    except Exception as e:
                        print(f"  {inp.name:30s} = (err: {e})")

    print(f"\n=== Material Output node connections ===")
    for n in m.node_tree.nodes:
        if n.bl_idname == "ShaderNodeOutputMaterial":
            for inp in n.inputs:
                if inp.is_linked:
                    src = inp.links[0].from_node
                    print(f"  {inp.name} ← {src.name!r} ({src.bl_idname})")
                else:
                    print(f"  {inp.name} = (unconnected)")
