"""Dump all window-related meshes' positions to figure out which one
the camera sees as 'bright window pixels' on the right side of the
image."""
import bpy

scene = bpy.context.scene
cam = scene.camera
print(f"Camera @ {tuple(round(c, 3) for c in cam.location)}")
m = cam.matrix_world
fwd = (-m.col[2][0], -m.col[2][1], -m.col[2][2])
right = (m.col[0][0], m.col[0][1], m.col[0][2])
up = (m.col[1][0], m.col[1][1], m.col[1][2])
print(f"  forward = {tuple(round(c, 3) for c in fwd)}")
print(f"  right   = {tuple(round(c, 3) for c in right)}")
print(f"  up      = {tuple(round(c, 3) for c in up)}")

print(f"\n=== window-related objects ===")
keywords = ("window", "Window", "portal")
for obj in scene.objects:
    if obj.type != "MESH":
        continue
    mats = [s.material.name if s.material else "?" for s in obj.material_slots]
    if not any(k in obj.name for k in keywords) and not any(
        any(k in m for k in ("portal", "Portal", "Glass", "glass")) for m in mats
    ):
        continue
    loc = tuple(round(c, 3) for c in obj.location)
    npoly = len(obj.data.polygons) if obj.data else 0
    bbox_min = obj.bound_box[0]
    bbox_max = obj.bound_box[6]
    bbox_w = obj.matrix_world @ obj.matrix_world.inverted() @ obj.matrix_world  # placeholder
    # Compute world-space extents
    import mathutils
    vs = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    minc = (min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs))
    maxc = (max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs))
    print(f"  {obj.name}: loc={loc}, polys={npoly}, mats={mats}")
    print(f"    bbox (world): min=({minc[0]:.2f},{minc[1]:.2f},{minc[2]:.2f}), "
          f"max=({maxc[0]:.2f},{maxc[1]:.2f},{maxc[2]:.2f})")
    print(f"    visible_camera={obj.visible_camera}, visible_diffuse={obj.visible_diffuse}, "
          f"hide_render={obj.hide_render}")
