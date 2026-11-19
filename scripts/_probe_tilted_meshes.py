"""List every visible MESH object whose local-space Z extent is
non-trivial (>= 0.01 m local). After hiding the Plane cavity mesh, the
slope wedge is still visible in vibrt's preview, so the offending
geometry must be one of these tilted/curved planes."""
import bpy
import bpy_extras

scene = bpy.context.scene
cam = scene.camera

print(f"camera at {tuple(round(c, 2) for c in cam.matrix_world.translation)}")
print()

results = []
for obj in scene.objects:
    if obj.type != "MESH":
        continue
    if obj.hide_render:
        continue
    me = obj.data
    if not me.vertices:
        continue
    zs = [v.co.z for v in me.vertices]
    z_var = max(zs) - min(zs)
    scale_z = obj.matrix_world.to_scale().z
    world_var = z_var * abs(scale_z)
    if world_var < 0.05:
        continue
    # World AABB
    import mathutils
    bbox_w = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    wxs = [p.x for p in bbox_w]; wys = [p.y for p in bbox_w]; wzs = [p.z for p in bbox_w]
    # How many bbox corners are in screen?
    in_screen = 0
    for p in bbox_w:
        co = bpy_extras.object_utils.world_to_camera_view(scene, cam, p)
        if 0 <= co.x <= 1 and 0 <= co.y <= 1 and co.z > 0:
            in_screen += 1
    results.append((world_var, obj.name, len(me.vertices), len(me.polygons),
                    min(wxs), max(wxs), min(wys), max(wys), min(wzs), max(wzs),
                    in_screen))

results.sort(reverse=True)
print(f"{'name':32s} {'verts':>6s} {'polys':>6s} {'world Zvar':>10s} "
      f"{'wAABB X':>16s} {'wAABB Y':>16s} {'wAABB Z':>16s} {'inScr':>6s}")
for r in results:
    wv, name, vc, pc, xmn, xmx, ymn, ymx, zmn, zmx, ins = r
    print(f"{name:32s} {vc:>6d} {pc:>6d} {wv:>10.3f} "
          f"{xmn:>+7.2f},{xmx:>+7.2f} {ymn:>+7.2f},{ymx:>+7.2f} "
          f"{zmn:>+7.2f},{zmx:>+7.2f} {ins:>6d}")
