"""Check world Volume socket and any volumetric materials in the scene.

Run via: blender -b <scene>.blend --python scripts/_probe_world_volume.py
"""
import bpy

scene = bpy.context.scene
world = scene.world

print(f"world: {world.name}")
print(f"world.use_nodes: {world.use_nodes}")

if world.use_nodes:
    out = next(
        (n for n in world.node_tree.nodes
         if n.bl_idname == "ShaderNodeOutputWorld"),
        None,
    )
    if out is not None:
        for sock_name in ("Surface", "Volume", "Displacement"):
            sock = out.inputs.get(sock_name)
            if sock is None:
                continue
            if sock.is_linked:
                from_node = sock.links[0].from_node
                print(f"  out.{sock_name} ← {from_node.bl_idname} '{from_node.name}'")
            else:
                print(f"  out.{sock_name}: (not linked)")

# Also enumerate any volumetric usage on materials.
print()
print("=== materials with Volume socket linked or transmission > 0 ===")
for mat in bpy.data.materials:
    if not mat.use_nodes or mat.node_tree is None:
        continue
    out = next(
        (n for n in mat.node_tree.nodes
         if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output),
        None,
    )
    if out is None:
        out = next(
            (n for n in mat.node_tree.nodes
             if n.bl_idname == "ShaderNodeOutputMaterial"),
            None,
        )
    if out is None:
        continue
    vol_sock = out.inputs.get("Volume")
    if vol_sock is not None and vol_sock.is_linked:
        from_node = vol_sock.links[0].from_node
        print(f"  {mat.name}: Volume ← {from_node.bl_idname} '{from_node.name}'")

# Compositor / film transparency / world mist?
print()
print("=== render-settings checks ===")
print(f"render.film_transparent = {scene.render.film_transparent}")
print(f"world.cycles_visibility (camera): {world.cycles_visibility.camera if hasattr(world, 'cycles_visibility') else 'n/a'}")

# Mist pass?
print(f"view_layer.use_pass_mist = {scene.view_layers[0].use_pass_mist}")

# Compositor nodes?
if scene.use_nodes and scene.node_tree is not None:
    print(f"compositor: {len(scene.node_tree.nodes)} nodes")
    for n in scene.node_tree.nodes:
        print(f"  {n.bl_idname} '{n.name}'  mute={n.mute}")
else:
    print("compositor: (disabled / no tree)")

# Cycles light sampling settings — maybe the world MIS sample weight differs.
print()
print("=== world cycles settings ===")
if hasattr(world, "cycles"):
    cs = world.cycles
    for attr in ("sampling_method", "sample_map_resolution", "max_bounces",
                 "homogeneous_volume", "volume_sampling", "volume_interpolation"):
        if hasattr(cs, attr):
            print(f"  world.cycles.{attr} = {getattr(cs, attr)}")

print()
print("=== scene.cycles ===")
sc = scene.cycles
for attr in ("samples", "preview_samples", "use_denoising", "max_bounces",
             "diffuse_bounces", "glossy_bounces", "transmission_bounces",
             "volume_bounces", "transparent_max_bounces",
             "use_adaptive_sampling", "blur_glossy", "sample_clamp_direct",
             "sample_clamp_indirect", "light_sampling_threshold"):
    if hasattr(sc, attr):
        print(f"  scene.cycles.{attr} = {getattr(sc, attr)}")
