"""Probe lone_monk to understand why vibrt is ~1/2 as bright as Cycles.
Look at the world / sky setup, render settings, and dominant
materials.
"""
import bpy

scene = bpy.context.scene
print("=== Render settings ===")
print(f"  resolution: {scene.render.resolution_x}x{scene.render.resolution_y} @ {scene.render.resolution_percentage}%")
cy = getattr(scene, "cycles", None)
if cy is not None:
    print(f"  cycles.samples: {cy.samples}")
    print(f"  max_bounces: {cy.max_bounces} (diffuse {cy.diffuse_bounces}, glossy {cy.glossy_bounces}, transmission {cy.transmission_bounces})")
    print(f"  sample_clamp_direct: {cy.sample_clamp_direct}, indirect: {cy.sample_clamp_indirect}")
    print(f"  blur_glossy (filter): {cy.blur_glossy}")
print(f"  view_transform: {scene.view_settings.view_transform}, look: {scene.view_settings.look}")
print(f"  exposure: {scene.view_settings.exposure}, gamma: {scene.view_settings.gamma}")

print("\n=== World ===")
w = scene.world
if w is not None:
    print(f"  {w.name}: use_nodes={w.use_nodes}")
    if w.use_nodes and w.node_tree is not None:
        for n in w.node_tree.nodes:
            extra = ""
            if n.bl_idname == "ShaderNodeBackground":
                ssock = n.inputs.get("Strength")
                if ssock is not None:
                    extra = f" Strength={ssock.default_value!r} linked={ssock.is_linked}"
            elif n.bl_idname == "ShaderNodeTexSky":
                extra = (f" sky_type={n.sky_type}"
                         f" sun_elevation={n.sun_elevation:.4f}"
                         f" sun_rotation={n.sun_rotation:.4f}"
                         f" turbidity={n.turbidity}"
                         f" altitude={n.altitude}"
                         f" sun_size={n.sun_size:.6f}"
                         f" sun_intensity={n.sun_intensity}"
                         f" sun_disc={n.sun_disc}")
            elif n.bl_idname == "ShaderNodeMath":
                extra = f" operation={n.operation}"
            elif n.bl_idname == "ShaderNodeMix" or n.bl_idname == "ShaderNodeMixRGB":
                if hasattr(n, "blend_type"):
                    extra = f" blend={n.blend_type}"
            print(f"    node {n.name!r} bl_idname={n.bl_idname}{extra}")
        print("  links:")
        for lk in w.node_tree.links:
            print(f"    {lk.from_node.name}.{lk.from_socket.name} -> {lk.to_node.name}.{lk.to_socket.name}")

print("\n=== Lights ===")
for obj in scene.objects:
    if obj.type != 'LIGHT':
        continue
    light = obj.data
    print(f"  [{obj.name}] type={light.type} color={tuple(round(v, 3) for v in light.color)} energy={light.energy}")

print("\n=== Top materials by polygon count ===")
mats_by_size = {}
for obj in scene.objects:
    if obj.type != 'MESH':
        continue
    me = obj.data
    poly_count = len(me.polygons)
    for slot in obj.material_slots:
        if slot.material is None:
            continue
        mats_by_size.setdefault(slot.material.name, []).append((obj.name, poly_count))
items = sorted(mats_by_size.items(), key=lambda kv: -sum(p for _, p in kv[1]))
for name, users in items[:8]:
    total_polys = sum(p for _, p in users)
    mat = bpy.data.materials.get(name)
    print(f"  {name!r}: {total_polys} polys total")
    if mat and mat.use_nodes and mat.node_tree:
        out = next((n for n in mat.node_tree.nodes
                    if n.bl_idname == "ShaderNodeOutputMaterial"
                    and n.is_active_output), None)
        if out is None:
            continue
        surf = out.inputs.get("Surface")
        if surf is None or not surf.is_linked:
            continue
        src = surf.links[0].from_node
        print(f"    surface={src.bl_idname}")
