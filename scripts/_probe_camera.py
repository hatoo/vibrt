"""Dump active camera + render settings for the loaded .blend.

Run via: blender -b <scene>.blend --python scripts/_probe_camera.py
"""

import math
import bpy

scene = bpy.context.scene
rd = scene.render
cam_obj = scene.camera
cam = cam_obj.data

w = int(rd.resolution_x * rd.resolution_percentage / 100.0)
h = int(rd.resolution_y * rd.resolution_percentage / 100.0)
aspect = (w * rd.pixel_aspect_x) / (h * rd.pixel_aspect_y) if h > 0 else 1.0

print(f"resolution: {w} x {h} (raw {rd.resolution_x}x{rd.resolution_y} @ {rd.resolution_percentage}%)")
print(f"pixel_aspect: {rd.pixel_aspect_x} x {rd.pixel_aspect_y}")
print(f"image aspect (w/h, with pixel_aspect): {aspect:.6f}")
print()
print(f"camera.type        = {cam.type}")
print(f"camera.sensor_fit  = {cam.sensor_fit}")
print(f"camera.sensor_width  = {cam.sensor_width}")
print(f"camera.sensor_height = {cam.sensor_height}")
print(f"camera.lens          = {cam.lens}")
print(f"camera.angle   = {math.degrees(cam.angle):.4f} deg")
print(f"camera.angle_x = {math.degrees(cam.angle_x):.4f} deg")
print(f"camera.angle_y = {math.degrees(cam.angle_y):.4f} deg")
print(f"camera.shift_x = {cam.shift_x}, shift_y = {cam.shift_y}")
print(f"camera.clip_start = {cam.clip_start}, clip_end = {cam.clip_end}")

# Authoritative frame from Blender. Returns 4 corners in CAMERA-LOCAL space at
# z = -clip_start. Order: top-right, bottom-right, bottom-left, top-left.
print()
print("=== view_frame(scene=scene) (camera-local) ===")
frame = cam.view_frame(scene=scene)
for i, p in enumerate(frame):
    print(f"  corner[{i}] = ({p.x:.6f}, {p.y:.6f}, {p.z:.6f})")

# Normalise to z = -1 plane.
print()
print("=== view_frame normalised to z=-1 ===")
norm = [(p.x / -p.z, p.y / -p.z) for p in frame]
for i, (x, y) in enumerate(norm):
    print(f"  norm[{i}] = ({x:.6f}, {y:.6f})")

# Half extents at z = -1, plus center (which encodes shift).
xs = [p[0] for p in norm]
ys = [p[1] for p in norm]
half_w = (max(xs) - min(xs)) * 0.5
half_h = (max(ys) - min(ys)) * 0.5
cx = (max(xs) + min(xs)) * 0.5
cy = (max(ys) + min(ys)) * 0.5
print()
print(f"frame half_w = {half_w:.6f}  → fov_x = {math.degrees(2*math.atan(half_w)):.4f} deg")
print(f"frame half_h = {half_h:.6f}  → fov_y = {math.degrees(2*math.atan(half_h)):.4f} deg")
print(f"frame center = ({cx:.6f}, {cy:.6f}) (offset from optical axis)")
print(f"shift in NDC: ({cx/half_w:.6f}, {cy/half_h:.6f})  (fraction of half-frame)")

# Compare to what the exporter currently sends:
print()
print("=== exporter would send (current code) ===")
print(f"  fov_y_rad     = {cam.angle_y:.6f}  ({math.degrees(cam.angle_y):.4f} deg)")
print(f"  shift_x       = {cam.shift_x}")
print(f"  shift_y       = {cam.shift_y}")
print()
print("=== what the device code does with that ===")
# Device: half_h = tan(fov_y/2); half_w = half_h * aspect;
# px_ndc_offset = 2 * shift_x; py_ndc_offset = 2 * shift_y
device_half_h = math.tan(cam.angle_y * 0.5)
device_half_w = device_half_h * aspect
print(f"  device half_h = {device_half_h:.6f} → fov_y = {math.degrees(2*math.atan(device_half_h)):.4f} deg")
print(f"  device half_w = {device_half_w:.6f} → fov_x = {math.degrees(2*math.atan(device_half_w)):.4f} deg")
print(f"  device shift in view-space at z=-1: x={2*cam.shift_x*device_half_w:.6f}, y={2*cam.shift_y*device_half_h:.6f}")
print()
print("=== authoritative shift in view-space at z=-1 ===")
print(f"  cx = {cx:.6f}, cy = {cy:.6f}")
print()
print("=== ratio: authoritative / device ===")
print(f"  half_w: {half_w/device_half_w:.6f}")
print(f"  half_h: {half_h/device_half_h:.6f}")
