"""Identify which object/material is visible at the hot pixels
(y=166, x=200) and (y=185, x=142) in the lone_monk db=0 render
(360x270). Uses ray_cast from the camera through the pixel center.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import bpy
from mathutils import Vector

scene = bpy.context.scene
cam_obj = scene.camera
cam = cam_obj.data
W, H = 360, 270  # PCT=25 of 1440x1080? we render 360x270
# Actually scene.render.resolution_x * pct/100 = 1440 * 0.25 = 360
# And resolution_y * pct/100 = 1080 * 0.25 = 270

# Compute camera ray for pixel (px, py) where (0,0) is bottom-left
# Blender pixel coords: foreach_get returns rows BOTTOM-UP
def pixel_to_dir(py, px, w, h):
    # py from script (0 = bottom), Blender's foreach_get y=0 is bottom row
    # But our analysis used numpy reshape((h,w,4)) where index 0 = bottom row.
    # So py=166 means 166 rows from bottom in a 270 image.
    # NDC: x = (px + 0.5)/w, y = (py + 0.5)/h, both in [0,1]
    nx = (px + 0.5) / w
    ny = (py + 0.5) / h
    # camera space: x=(2nx-1)*tan(fov/2)*aspect, y=(2ny-1)*tan(fov/2)
    # cam.angle is the diagonal/horizontal/vertical depending on cam.sensor_fit
    if cam.sensor_fit == "VERTICAL":
        ty = math.tan(cam.angle / 2)
        tx = ty * (w / h)
    else:  # HORIZONTAL or AUTO
        tx = math.tan(cam.angle / 2)
        ty = tx * (h / w)
    cx = (2 * nx - 1) * tx
    cy = (2 * ny - 1) * ty
    # Camera looks down -Z in local space
    local_dir = Vector((cx, cy, -1)).normalized()
    world_dir = (cam_obj.matrix_world.to_3x3() @ local_dir).normalized()
    return cam_obj.matrix_world.translation, world_dir


def trace(label, py, px):
    origin, direction = pixel_to_dir(py, px, W, H)
    print(f"\n=== {label} pixel y={py} x={px} ===")
    print(f"  origin   = ({origin.x:.3f}, {origin.y:.3f}, {origin.z:.3f})")
    print(f"  direction= ({direction.x:.3f}, {direction.y:.3f}, {direction.z:.3f})")

    # ray_cast on the depsgraph
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result = scene.ray_cast(depsgraph, origin, direction)
    success, location, normal, index, obj, matrix = result
    if not success:
        print(f"  → no hit (escapes to sky)")
        return
    print(f"  → hit obj   = {obj.name!r}")
    print(f"    location  = ({location.x:.3f}, {location.y:.3f}, {location.z:.3f})")
    print(f"    distance  = {(location - origin).length:.3f} m")
    print(f"    normal    = ({normal.x:.3f}, {normal.y:.3f}, {normal.z:.3f})")
    print(f"    poly idx  = {index}")
    if obj.type == "MESH":
        # Find material of that polygon
        try:
            poly = obj.data.polygons[index]
            mat_idx = poly.material_index
            if mat_idx < len(obj.material_slots):
                ms = obj.material_slots[mat_idx]
                print(f"    material  = {ms.material.name if ms.material else '<none>'}")
        except Exception as e:
            print(f"    material  = (err {e})")


# Sun direction for reference
sun_dir = (0.301, -0.521, +0.799)
print(f"sun direction (toward) = {sun_dir}")
print(f"camera location = {tuple(round(x, 3) for x in cam_obj.matrix_world.translation)}")

trace("hot1 (vibrt 35,17,10  cycles 8,6,5)", 166, 200)
trace("hot2 (vibrt 30,18,14  cycles 8,5,4)", 185, 142)
