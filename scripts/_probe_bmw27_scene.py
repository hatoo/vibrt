"""Probe bmw27 to understand why vibrt renders +0.10 brighter than
Cycles. Lists lights, world, materials, render settings — anything that
could cause an over-bright bias.
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
            elif n.bl_idname == "ShaderNodeTexEnvironment":
                extra = f" image={n.image.name if n.image else None!r}"
            print(f"    node {n.name!r} bl_idname={n.bl_idname}{extra}")

print("\n=== Lights ===")
for obj in scene.objects:
    if obj.type != 'LIGHT':
        continue
    light = obj.data
    print(f"  [{obj.name}] type={light.type} color={tuple(round(v, 3) for v in light.color)} energy={light.energy}")

print("\n=== Materials (top contributors) ===")
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
for name, users in sorted(mats_by_size.items()):
    total_polys = sum(p for _, p in users)
    if total_polys < 100:
        continue
    mat = bpy.data.materials.get(name)
    print(f"  {name!r}: {total_polys} polys total, {len(users)} object slot(s)")
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
        if src.bl_idname == "ShaderNodeBsdfPrincipled":
            for k in ("Base Color", "Metallic", "Roughness", "IOR", "Transmission Weight",
                       "Specular IOR Level", "Coat Weight", "Sheen Weight"):
                s = src.inputs.get(k)
                if s is None:
                    continue
                v = s.default_value
                if hasattr(v, "__len__"):
                    v = tuple(round(c, 3) for c in v)
                print(f"      {k}: {v} linked={s.is_linked}")
