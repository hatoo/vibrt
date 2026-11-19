"""Inspect every mesh that uses the 'Lamp Gubi' material — how many triangles,
how big, and whether the emissive-quad promotion would catch it."""
import bpy
from mathutils import Vector

target_mat_names = {"Lamp Gubi"}

for obj in bpy.context.scene.objects:
    if obj.type != 'MESH':
        continue
    mat_names = {s.material.name for s in obj.material_slots if s.material}
    if not mat_names & target_mat_names:
        continue
    me = obj.data
    print(f"=== {obj.name} ===")
    print(f"  matrix.translation = ({obj.matrix_world.translation.x:.2f}, "
          f"{obj.matrix_world.translation.y:.2f}, {obj.matrix_world.translation.z:.2f})")
    print(f"  vertices = {len(me.vertices)}  polygons = {len(me.polygons)}")
    print(f"  material slots = {[s.material.name if s.material else None for s in obj.material_slots]}")
    # Per-poly material indices
    if me.polygons:
        mi_set = set(p.material_index for p in me.polygons)
        print(f"  material_indices used = {mi_set}")
        for mi in mi_set:
            slot = obj.material_slots[mi].material if mi < len(obj.material_slots) else None
            n_polys = sum(1 for p in me.polygons if p.material_index == mi)
            n_tris = sum(len(p.vertices) - 2 for p in me.polygons if p.material_index == mi)
            print(f"    slot {mi}: '{slot.name if slot else None}' — {n_polys} polygons / {n_tris} triangles")
    # Bounding box size
    bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [v.x for v in bb]; ys = [v.y for v in bb]; zs = [v.z for v in bb]
    print(f"  bbox size: ({max(xs)-min(xs):.3f}, {max(ys)-min(ys):.3f}, {max(zs)-min(zs):.3f})")
