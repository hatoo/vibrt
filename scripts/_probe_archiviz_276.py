"""Find what is at world (4.65, 8.94, 1.11) — the t=2.76m hit on the TL_corner
camera ray after clip_start=2.5. Use a fine ray_cast with many hop tries to
catch hair/particle/instanced geometry."""
import bpy
from mathutils import Vector

scene = bpy.context.scene
deps = bpy.context.evaluated_depsgraph_get()
target = Vector((4.65, 8.94, 1.11))

# Find every object whose evaluated bbox contains the target.
print(f"target: {tuple(target)}")
hits = []
for obj in scene.objects:
    if obj.type not in ('MESH', 'CURVES', 'POINTCLOUD'):
        continue
    eval_obj = obj.evaluated_get(deps)
    bb = [eval_obj.matrix_world @ Vector(c) for c in eval_obj.bound_box]
    xs = [v.x for v in bb]; ys = [v.y for v in bb]; zs = [v.z for v in bb]
    if (min(xs)-0.05 <= target.x <= max(xs)+0.05
            and min(ys)-0.05 <= target.y <= max(ys)+0.05
            and min(zs)-0.05 <= target.z <= max(zs)+0.05):
        hits.append((obj.type, obj.name, len(eval_obj.data.vertices) if obj.type == 'MESH' else -1))

# Also look at every object instance from the depsgraph (catches particles/instances).
print("=== depsgraph instances near target ===")
for inst in deps.object_instances:
    if inst.object.type not in ('MESH',):
        continue
    o = inst.matrix_world @ Vector((0, 0, 0))
    if (o - target).length < 1.0:
        print(f"  {'INST' if inst.is_instance else 'real'}  obj={inst.object.name}  origin=({o.x:.2f},{o.y:.2f},{o.z:.2f})")

print(f"\n=== bbox-contains-target ({len(hits)}) ===")
for t in hits:
    print(f"  {t}")

# Try ray_cast from target slightly displaced (since target IS very close to wall back-face).
eye = scene.camera.matrix_world.translation
direction = (target - eye).normalized()
# Cast from t=2.55 forward (just past clip 2.5)
origin = eye + direction * 2.55
print(f"\n=== ray_cast from origin past clip_start ===")
for tries in range(8):
    res, loc, n, idx, obj, _ = scene.ray_cast(deps, origin, direction, distance=2.0)
    if not res:
        print("  no more hits")
        break
    t = (loc - eye).length
    print(f"  t={t:.3f} obj={obj.name:30s} loc=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f}) n=({n.x:+.2f},{n.y:+.2f},{n.z:+.2f})")
    origin = loc + direction * 1e-4
