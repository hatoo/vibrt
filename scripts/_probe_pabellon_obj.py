"""Find all objects in pabellon and which materials they use, including
particle hairs and instancer collections."""
import bpy

print("=== all mesh objects with materials ===")
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    mats = [s.material.name if s.material else "(none)" for s in obj.material_slots]
    if not mats:
        continue
    print(f"  {obj.name:50s} [{obj.type}] mats={mats}")

print("\n=== particle systems ===")
for obj in bpy.data.objects:
    if not obj.particle_systems:
        continue
    for psys in obj.particle_systems:
        ps = psys.settings
        print(f"  {obj.name} -> psys '{psys.name}' settings='{ps.name}' "
              f"render_type={ps.render_type} instance_object={ps.instance_object.name if ps.instance_object else None} "
              f"instance_collection={ps.instance_collection.name if ps.instance_collection else None}")

print("\n=== collections (for instancers) ===")
for c in bpy.data.collections:
    print(f"  '{c.name}' objects={[o.name for o in c.objects[:5]]}{'...' if len(c.objects) > 5 else ''}")
