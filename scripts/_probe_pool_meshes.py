"""Inspect every object whose name suggests pool / water / basin and
dump its mesh: vertex count, world-space AABB, per-axis range, modifier
stack (so we can spot a subdivision-surface / displacement / shrinkwrap
that vibrt isn't applying), and parent rig."""
import bpy

scene = bpy.context.scene
keywords = ("pool", "water", "basin", "lotus", "pond", "pebble", "plane")

for obj in scene.objects:
    if obj.type != "MESH":
        continue
    n = obj.name.lower()
    if not any(k in n for k in keywords):
        continue
    me = obj.data
    print(f"\n=== {obj.name} (data='{me.name}', verts={len(me.vertices)}, "
          f"polys={len(me.polygons)}) ===")
    print(f"  hide_render={obj.hide_render} hide_viewport={obj.hide_viewport}")
    print(f"  parent={obj.parent.name if obj.parent else None}")
    print(f"  matrix_world translation={tuple(round(c, 3) for c in obj.matrix_world.translation)}")
    print(f"  scale={tuple(round(c, 3) for c in obj.matrix_world.to_scale())}")
    rot_euler = obj.matrix_world.to_euler()
    print(f"  rot_euler={tuple(round(c, 4) for c in rot_euler)}")
    # World-space bounding box
    bbox_world = [obj.matrix_world @ v for v in (
        __import__("mathutils").Vector(c) for c in obj.bound_box
    )]
    xs = [v.x for v in bbox_world]
    ys = [v.y for v in bbox_world]
    zs = [v.z for v in bbox_world]
    print(f"  world AABB: x=[{min(xs):.3f},{max(xs):.3f}] y=[{min(ys):.3f},{max(ys):.3f}] z=[{min(zs):.3f},{max(zs):.3f}]")
    # Local vertex z-range (to detect tilted planes)
    if me.vertices:
        local_zs = [v.co.z for v in me.vertices]
        local_xs = [v.co.x for v in me.vertices]
        local_ys = [v.co.y for v in me.vertices]
        print(f"  local Z range: [{min(local_zs):.4f}, {max(local_zs):.4f}] (var={max(local_zs)-min(local_zs):.4f})")
        print(f"  local X range: [{min(local_xs):.4f}, {max(local_xs):.4f}]")
        print(f"  local Y range: [{min(local_ys):.4f}, {max(local_ys):.4f}]")
    if obj.modifiers:
        print(f"  modifiers:")
        for m in obj.modifiers:
            extra = ""
            if m.type == "DISPLACE":
                extra = f" strength={m.strength} mid_level={m.mid_level} texture={m.texture.name if m.texture else None}"
            if m.type == "SUBSURF":
                extra = f" levels={m.levels}/{m.render_levels}"
            print(f"    {m.type} '{m.name}' show_render={m.show_render} show_viewport={m.show_viewport}{extra}")
    # Check for shape keys
    if me.shape_keys:
        print(f"  shape_keys: {[(k.name, round(k.value, 3)) for k in me.shape_keys.key_blocks]}")
    # Materials
    mats = [s.material.name if s.material else None for s in obj.material_slots]
    print(f"  materials: {mats}")
