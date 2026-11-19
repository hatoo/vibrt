"""Dump every lamp + IES Texture node setup in the ies_light test scene
so we can compare the warm color Cycles renders vs vibrt's neutral output."""
import bpy

print(f"=== Lamps in scene ===")
for obj in bpy.data.objects:
    if obj.type != "LIGHT":
        continue
    light = obj.data
    print(f"\n  {obj.name!r}: type={light.type}")
    print(f"    color = {list(light.color)}")
    print(f"    energy = {light.energy}")
    if hasattr(light, "use_nodes"):
        print(f"    use_nodes = {light.use_nodes}")
    if light.use_nodes and light.node_tree:
        out = None
        for n in light.node_tree.nodes:
            if n.bl_idname == "ShaderNodeOutputLight":
                out = n
                break
        if out:
            surf = out.inputs.get("Surface")
            if surf and surf.is_linked:
                src = surf.links[0].from_node
                print(f"    Output.Surface ← {src.name!r} ({src.bl_idname})")
                # Walk full tree
                def walk(n, depth=0):
                    ind = "      " + "  " * depth
                    print(f"{ind}{n.bl_idname}: {n.name!r}")
                    for inp in n.inputs:
                        if inp.is_linked:
                            from_n = inp.links[0].from_node
                            from_s = inp.links[0].from_socket.name
                            print(f"{ind}  in {inp.name} ← {from_n.name!r}.{from_s}")
                            if depth < 5:
                                walk(from_n, depth + 1)
                        else:
                            try:
                                v = inp.default_value
                                if hasattr(v, "__iter__"):
                                    v = list(v)
                                print(f"{ind}  in {inp.name} = {v}")
                            except Exception:
                                pass
                walk(src)
            else:
                print(f"    Output.Surface = (unconnected)")

# Check world too — maybe the warm color is from the world background
print(f"\n=== World ===")
w = bpy.context.scene.world
if w:
    print(f"  name: {w.name}")
    print(f"  color: {list(w.color)}")
    if w.use_nodes and w.node_tree:
        for n in w.node_tree.nodes:
            if n.bl_idname == "ShaderNodeOutputWorld":
                surf = n.inputs.get("Surface")
                if surf and surf.is_linked:
                    src = surf.links[0].from_node
                    print(f"  Output.Surface ← {src.name!r} ({src.bl_idname})")
                    for inp in src.inputs:
                        if not inp.is_linked:
                            try:
                                v = inp.default_value
                                if hasattr(v, "__iter__"):
                                    v = list(v)
                                print(f"    {inp.name} = {v}")
                            except Exception:
                                pass

# Check view transform
print(f"\n=== Render settings ===")
print(f"  view_transform: {bpy.context.scene.view_settings.view_transform}")
print(f"  look:           {bpy.context.scene.view_settings.look}")
print(f"  exposure:       {bpy.context.scene.view_settings.exposure}")
print(f"  gamma:          {bpy.context.scene.view_settings.gamma}")
