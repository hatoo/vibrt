"""Inspect particle systems on the lily / lotus / pebbles scatter planes."""
import bpy

scene = bpy.context.scene
for name in ("lotus_scattering_plane", "pebbles_scatter", "tree_scatter_plane"):
    obj = scene.objects.get(name)
    if obj is None:
        print(f"{name}: not found")
        continue
    print(f"\n=== {name} ===")
    print(f"  type: {obj.type}")
    print(f"  hide_render: {obj.hide_render}")
    print(f"  visible_camera: {getattr(obj, 'visible_camera', '?')}")
    for a in ("show_instancer_for_render", "show_instancer_for_viewport"):
        if hasattr(obj, a):
            print(f"  obj.{a} = {getattr(obj, a)}")
    print(f"  material slots: {[s.material.name if s.material else '(empty)' for s in obj.material_slots]}")
    if obj.particle_systems:
        for ps in obj.particle_systems:
            ps_settings = ps.settings
            print(f"  particle '{ps.name}':")
            print(f"    render_type = {ps_settings.render_type}")
            print(f"    instance_object = {ps_settings.instance_object.name if ps_settings.instance_object else '(none)'}")
            print(f"    instance_collection = {ps_settings.instance_collection.name if ps_settings.instance_collection else '(none)'}")
            for a in ("use_render_emitter", "show_instancer_for_render"):
                if hasattr(ps_settings, a):
                    print(f"    settings.{a} = {getattr(ps_settings, a)}")
            print(f"    count = {ps_settings.count}, type = {ps_settings.type}")
    else:
        print(f"  no particle systems")
    if obj.modifiers:
        for m in obj.modifiers:
            print(f"  modifier '{m.name}' type={m.type}")
