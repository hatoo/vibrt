"""Print the flat-archiviz scene camera lens settings (lens type, sensor fit,
shift, focal/fov) and the render resolution so we can figure out why vibrt
is producing a circular black frame around the image."""
import bpy

scene = bpy.context.scene
cam_obj = scene.camera
cam = cam_obj.data

print(f"camera obj: {cam_obj.name}  (data='{cam.name}')")
print(f"  type           = {cam.type}")
print(f"  lens (mm)      = {cam.lens}")
print(f"  lens_unit      = {cam.lens_unit}")
print(f"  angle (rad)    = {cam.angle}")
print(f"  sensor_fit     = {cam.sensor_fit}")
print(f"  sensor_width   = {cam.sensor_width}")
print(f"  sensor_height  = {cam.sensor_height}")
print(f"  shift_x/shift_y= {cam.shift_x}, {cam.shift_y}")
print(f"  clip_start/end = {cam.clip_start}, {cam.clip_end}")
print(f"  ortho_scale    = {cam.ortho_scale}")
if hasattr(cam, "panorama_type"):
    print(f"  panorama_type  = {cam.panorama_type}")
if hasattr(cam, "fisheye_lens"):
    print(f"  fisheye_lens   = {cam.fisheye_lens}")
if hasattr(cam, "fisheye_fov"):
    print(f"  fisheye_fov    = {cam.fisheye_fov}")
if hasattr(cam, "dof"):
    print(f"  dof.use        = {cam.dof.use_dof}")
    print(f"  dof.aperture_fstop = {cam.dof.aperture_fstop}")

print()
print(f"render.resolution_x/y      = {scene.render.resolution_x}, {scene.render.resolution_y}")
print(f"render.resolution_percentage = {scene.render.resolution_percentage}")
print(f"render.pixel_aspect_x/y    = {scene.render.pixel_aspect_x}, {scene.render.pixel_aspect_y}")
print(f"render.use_border          = {scene.render.use_border}")
print(f"render.use_crop_to_border  = {scene.render.use_crop_to_border}")
print(f"render.border_min_x/max_x  = {scene.render.border_min_x}, {scene.render.border_max_x}")
print(f"render.border_min_y/max_y  = {scene.render.border_min_y}, {scene.render.border_max_y}")
