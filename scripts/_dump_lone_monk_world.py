"""Dump full lone_monk World node tree, including TexSky internals,
to verify there's no hidden multiplier (Strength != 1, MixShader, math
node multiply etc.).
"""
import bpy

scene = bpy.context.scene
world = scene.world
print(f"World: {world.name!r}")
print(f"  use_nodes = {world.use_nodes}")

if world.node_tree:
    print(f"  --- ALL NODES ---")
    for n in world.node_tree.nodes:
        print(f"  {n.bl_idname}/{n.name}")
        for inp in n.inputs:
            try:
                if inp.is_linked:
                    src = inp.links[0].from_node
                    sock = inp.links[0].from_socket.name
                    print(f"    in {inp.name} ← {src.bl_idname}/{src.name}.{sock}")
                else:
                    v = inp.default_value
                    if hasattr(v, "__iter__"):
                        v = tuple(round(float(x), 5) for x in v)
                    else:
                        v = round(float(v), 5) if isinstance(v, (int, float)) else v
                    print(f"    in {inp.name} = {v}")
            except Exception as ex:
                print(f"    in {inp.name} (err: {ex})")

# Find the TexSky node and dump all its props
sky = next((n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeTexSky"), None)
if sky:
    print(f"\n  --- TexSky props ---")
    for attr in ("sky_type", "sun_disc", "sun_size", "sun_intensity",
                 "sun_elevation", "sun_rotation", "altitude", "air_density",
                 "aerosol_density", "ozone_density", "ground_albedo", "turbidity"):
        if hasattr(sky, attr):
            print(f"  {attr} = {getattr(sky, attr)}")
