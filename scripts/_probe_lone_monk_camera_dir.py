"""Compute the world-space ray direction for cycles' brightest pixel.
"""
import bpy
import math

scene = bpy.context.scene
cam = scene.camera
cd = cam.data
print(f"camera location: {tuple(round(c, 3) for c in cam.location)}")
print(f"camera lens = {cd.lens}, sensor_width = {cd.sensor_width}, sensor_height = {cd.sensor_height}")
print(f"camera fov_x: {math.degrees(cd.angle_x):.1f}°, fov_y: {math.degrees(cd.angle_y):.1f}°")
m = cam.matrix_world
print(f"matrix_world cols:")
for i in range(4):
    print(f"  col[{i}] = {tuple(round(c, 3) for c in m.col[i])}")

# Camera frame in world space
right = m.col[0]
up = m.col[1]
forward = -m.col[2]
print(f"camera frame:")
print(f"  right   = ({right.x:.3f}, {right.y:.3f}, {right.z:.3f})")
print(f"  up      = ({up.x:.3f}, {up.y:.3f}, {up.z:.3f})")
print(f"  forward = ({forward.x:.3f}, {forward.y:.3f}, {forward.z:.3f})")

# For pixel (yi, xi) in image of size (h, w), Blender renders image
# top-down (foreach_get returns bottom-up). So pixel (yi=58, xi=0) in
# the foreach_get array = (h-1-58, 0) in image space top-down.
# Image is 360×270 (w=360, h=270).
w, h = 360, 270
# Brightest pixel in cycles: foreach_get index (58, 0).
# Convert to top-down: y_top = h-1-58 = 211. x = 0.
y_fg, x_fg = 58, 0
y_td = h - 1 - y_fg
x_td = x_fg
print(f"\ncycles brightest pixel: foreach_get index ({y_fg}, {x_fg}) = top-down ({y_td}, {x_td})")

# Compute ray direction in camera local frame.
# Local convention: -Z forward, +Y up, +X right.
# px = (x/w - 0.5) * 2 * tan(fov_x/2)
# py = (0.5 - y/h) * 2 * tan(fov_y/2)  // top-down y
# local direction = (px, py, -1)
import math
px = (x_td / w - 0.5) * 2 * math.tan(cd.angle_x / 2)
py = (0.5 - y_td / h) * 2 * math.tan(cd.angle_y / 2)
local_dir = (px, py, -1.0)
norm = math.sqrt(local_dir[0]**2 + local_dir[1]**2 + local_dir[2]**2)
local_dir = (local_dir[0] / norm, local_dir[1] / norm, local_dir[2] / norm)
print(f"local ray dir = ({local_dir[0]:.3f}, {local_dir[1]:.3f}, {local_dir[2]:.3f})")

# Convert to world: world_dir = px * right + py * up + (-1) * (-forward) = px*right + py*up + forward
import mathutils
wd = (local_dir[0] * mathutils.Vector(right[:3])
      + local_dir[1] * mathutils.Vector(up[:3])
      - local_dir[2] * mathutils.Vector((-forward.x, -forward.y, -forward.z)))
# Wait, that's not right. Let me redo: world_dir = R * local_dir where R is the matrix
# whose columns are right, up, -forward (since local -Z = world forward).
# Actually camera_to_world maps local Z-down to world forward. local_dir.z = -1 →
# world component = +forward.
wd = (local_dir[0] * mathutils.Vector(right[:3])
      + local_dir[1] * mathutils.Vector(up[:3])
      + local_dir[2] * mathutils.Vector((-forward.x, -forward.y, -forward.z)))
wd_n = wd.normalized()
print(f"world ray dir = ({wd_n.x:.4f}, {wd_n.y:.4f}, {wd_n.z:.4f})")

# Sun direction (toward sun) in lone_monk: from sky controls
# sun_elevation = 53° = 0.925 rad, sun_rotation = 150° = 2.618 rad
# Cycles convention: (cos(elev) sin(rot), cos(elev) cos(rot), sin(elev))
elev = 0.925
rot = 2.618
sun_dir = (math.cos(elev) * math.sin(rot), math.cos(elev) * math.cos(rot), math.sin(elev))
print(f"sun direction (toward sun) = ({sun_dir[0]:.4f}, {sun_dir[1]:.4f}, {sun_dir[2]:.4f})")

# Dot product
dp = wd_n.x * sun_dir[0] + wd_n.y * sun_dir[1] + wd_n.z * sun_dir[2]
angle_deg = math.degrees(math.acos(max(min(dp, 1), -1)))
sun_size_deg = math.degrees(0.0095)
print(f"\nangle from camera ray to sun direction: {angle_deg:.2f}° (sun_size = {sun_size_deg:.3f}°)")
print(f"  → ray {'INSIDE' if angle_deg < sun_size_deg / 2 else 'OUTSIDE'} sun cone")
