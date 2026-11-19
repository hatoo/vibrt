"""Print camera position + look direction, and project key pool meshes
into camera screen space so we can pinpoint which mesh produces the
unwanted 'slope' in the bottom-right of the vibrt preview."""
import bpy
import bpy_extras
import mathutils

scene = bpy.context.scene
cam = scene.camera
print(f"=== camera '{cam.name}' ===")
print(f"  loc = {tuple(round(x,3) for x in cam.matrix_world.translation)}")
fwd = (cam.matrix_world @ mathutils.Vector((0, 0, -1, 1))).xyz - cam.matrix_world.translation
print(f"  forward = {tuple(round(x,3) for x in fwd)}")
print(f"  res = {scene.render.resolution_x}x{scene.render.resolution_y}")

deps = bpy.context.evaluated_depsgraph_get()


def project(p):
    co = bpy_extras.object_utils.world_to_camera_view(scene, cam, p)
    return (round(co.x, 3), round(co.y, 3), round(co.z, 2))


print()
for name in ("Plane", "Plane.003", "water_plane_still", "lotus_scattering_plane",
             "pebbles_scatter"):
    obj = bpy.data.objects.get(name)
    if obj is None:
        continue
    print(f"=== {name} ===")
    mw = obj.matrix_world
    # World AABB corners (8) -> projected.
    bbox_w = [mw @ mathutils.Vector(c) for c in obj.bound_box]
    for i, p in enumerate(bbox_w):
        scr = project(p)
        flag = "  IN" if 0 <= scr[0] <= 1 and 0 <= scr[1] <= 1 and scr[2] > 0 else "  out"
        print(f"  bbox[{i}] world=({p.x:+7.2f},{p.y:+7.2f},{p.z:+6.2f}) "
              f"screen=({scr[0]:+.3f},{scr[1]:+.3f},z={scr[2]:+.2f}){flag}")
