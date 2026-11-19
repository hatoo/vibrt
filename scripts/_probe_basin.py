"""Find the pool basin object/material and dump it."""
import bpy

# What's at the pool bottom? Find objects with z near 0.8 (above pebbles_scatter at 0.803)
print("=== objects near pool floor (z=0.8 ish) ===")
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    z = obj.location.z
    if 0.0 < z < 1.3:
        # bounds via bbox
        zs = [(obj.matrix_world @ type(obj.location)(c)).z for c in obj.bound_box]
        if min(zs) < 1.3:
            mats = [s.material.name if s.material else None for s in obj.material_slots]
            print(f"  {obj.name:40s} z={z:.3f} z_range=({min(zs):.3f},{max(zs):.3f}) mats={mats}")

print("\n=== particle_count / display ===")
for name in ("pebbles_scatter",):
    obj = bpy.data.objects.get(name)
    if obj:
        for ps in obj.particle_systems:
            sett = ps.settings
            print(f"  {name}: count={sett.count} display_step={sett.display_step} "
                  f"render_step={sett.render_step} particle_size={sett.particle_size} "
                  f"size_random={sett.size_random}")

# All materials with name containing 'pool', 'basin', 'concrete', 'floor'
print("\n=== candidate basin materials ===")
for m in bpy.data.materials:
    if any(k in m.name.lower() for k in ("pool", "basin", "concrete", "floor", "bottom")):
        print(f"  '{m.name}'")
