"""Find every mesh polygon within ~2m of the camera. Looking for a near-camera
emissive helper / fill plane that flat-archiviz uses + clip_start=2 to hide
from the primary view."""
import bpy
from mathutils import Vector

scene = bpy.context.scene
deps = bpy.context.evaluated_depsgraph_get()
eye = scene.camera.matrix_world.translation

# Walk every evaluated mesh (incl. instances) and report polys near the eye.
all_close = []
for inst in deps.object_instances:
    obj = inst.object
    if obj.type != 'MESH':
        continue
    me = obj.data
    M = inst.matrix_world
    centroid_local = sum((v.co for v in me.vertices), Vector()) / max(1, len(me.vertices))
    centroid = M @ centroid_local
    dist = (centroid - eye).length
    if dist > 2.5:
        continue
    n_verts = len(me.vertices)
    n_polys = len(me.polygons)
    # Find the closest poly center
    closest = None
    for poly in me.polygons:
        c_local = sum((me.vertices[i].co for i in poly.vertices), Vector()) / max(1, len(poly.vertices))
        c = M @ c_local
        d = (c - eye).length
        if closest is None or d < closest[0]:
            closest = (d, c, poly.material_index)
    if closest:
        d, c, mi = closest
        slot_obj = inst.object
        mat = (slot_obj.material_slots[mi].material.name
               if mi < len(slot_obj.material_slots) and slot_obj.material_slots[mi].material else "(none)")
        all_close.append((d, obj.name, mat, n_verts, n_polys, c, inst.is_instance))

all_close.sort()
print("=== meshes/instances with closest poly within 2.5m of camera ===")
print(f"{'dist':>5}  {'obj':35s} {'closest_mat':25s} {'verts':>6} {'polys':>6}  pos")
for d, name, mat, nv, np_, c, isinst in all_close[:30]:
    tag = "INST" if isinst else "real"
    print(f"  {d:5.2f}  {name:35s} {mat:25s} {nv:6d} {np_:6d}  ({c.x:+.2f},{c.y:+.2f},{c.z:+.2f}) {tag}")
