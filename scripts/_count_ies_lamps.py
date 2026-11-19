"""Count and list ALL lamp objects (any depsgraph state) in ies_light."""
import bpy

print(f"=== bpy.data.objects (ALL) lamps ===")
for obj in bpy.data.objects:
    if obj.type == "LIGHT":
        print(f"  {obj.name!r}: type={obj.data.type}, hide_render={obj.hide_render}, "
              f"loc=({obj.location.x:.2f}, {obj.location.y:.2f}, {obj.location.z:.2f}), "
              f"in_scene={obj.name in bpy.context.scene.collection.all_objects}, "
              f"vis_geo={obj.visible_get()}")

print(f"\n=== depsgraph.object_instances (what exporter sees) ===")
depsgraph = bpy.context.evaluated_depsgraph_get()
n = 0
for inst in depsgraph.object_instances:
    obj = inst.object
    if obj is None:
        continue
    if obj.type == "LIGHT":
        n += 1
        light = obj.data
        print(f"  #{n}: {obj.name!r}: type={light.type}, color={list(light.color)}, "
              f"energy={light.energy}")
print(f"  total: {n}")
