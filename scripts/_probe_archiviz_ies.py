"""Inspect the TexIES nodes in flat-archiviz: list every property and check
whether IES data is cached in-blend (.ies text datablock) or only in the
external file."""
import bpy

# Check Texts datablocks for parsed IES data
print("=== bpy.data.texts ===")
for t in bpy.data.texts:
    print(f"  {t.name!r}  size={len(t.as_string())} chars  filepath={t.filepath!r}")

# Walk every TexIES node
print("\n=== TexIES nodes ===")
for obj in bpy.context.scene.objects:
    if obj.type != 'LIGHT':
        continue
    light = obj.data
    if not light.use_nodes or light.node_tree is None:
        continue
    for n in light.node_tree.nodes:
        if n.bl_idname != 'ShaderNodeTexIES':
            continue
        print(f"  [{obj.name}] node '{n.name}'")
        for prop in n.bl_rna.properties:
            if prop.identifier in ('rna_type', 'inputs', 'outputs', 'internal_links',
                                    'parent', 'select', 'show_options', 'show_preview',
                                    'show_texture', 'use_clamp', 'use_custom_color',
                                    'color', 'location', 'width', 'width_hidden', 'height',
                                    'name', 'label', 'type', 'mute', 'hide', 'bl_idname',
                                    'bl_label', 'bl_rna', 'bl_static_type', 'bl_description',
                                    'bl_icon', 'bl_height_default', 'bl_height_min',
                                    'bl_height_max', 'bl_width_default', 'bl_width_min',
                                    'bl_width_max'):
                continue
            try:
                val = getattr(n, prop.identifier)
                if hasattr(val, "__len__") and not isinstance(val, str):
                    val = list(val)[:5]
                print(f"    {prop.identifier} = {val!r}")
            except Exception as e:
                print(f"    {prop.identifier} = (error: {e})")
        # Also try to access n.ies as Text data
        if hasattr(n, 'ies') and n.ies is not None:
            print(f"    ies (Text datablock) name={n.ies.name!r} "
                  f"size={len(n.ies.as_string())} chars")
            print(f"    first 200 chars:\n{n.ies.as_string()[:200]}")
