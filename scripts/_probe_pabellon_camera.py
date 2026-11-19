"""Print the pabellon scene camera transform + ray direction so we can
match it against the equirect bake."""
import bpy
import math

scene = bpy.context.scene
cam = scene.camera
print(f"camera obj: {cam.name}")
print(f"  location = {tuple(round(x,3) for x in cam.location)}")
print(f"  rotation_euler = {tuple(round(x,3) for x in cam.rotation_euler)} (rad)")
print(f"  rotation_euler deg = {tuple(round(math.degrees(x),1) for x in cam.rotation_euler)}")

mat = cam.matrix_world
forward = -mat.col[2].xyz  # -Z is camera forward
up = mat.col[1].xyz
right = mat.col[0].xyz
print(f"  world forward (-Z) = {tuple(round(x,3) for x in forward)}")
print(f"  world up      (+Y) = {tuple(round(x,3) for x in up)}")
print(f"  world right   (+X) = {tuple(round(x,3) for x in right)}")

# Spherical coords of camera-forward
fx, fy, fz = forward
theta = math.degrees(math.acos(max(-1, min(1, fz))))     # 0 at zenith
phi = math.degrees(math.atan2(fy, fx))                    # vibrt's `phi = atan2(dir.y, dir.x)`
print(f"  forward as spherical: theta_from_zenith={theta:.1f}deg phi_atan2_yx={phi:.1f}deg")
