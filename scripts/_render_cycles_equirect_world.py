"""Render the world via a panoramic equirect camera in Cycles. This
gives us the ground-truth `world(dir)` function — the radiance Cycles
emits per direction, sampled at every (theta, phi). Compare against
vibrt's host-rasterised Mixed grid to localise the bug."""
import bpy, os

scene = bpy.context.scene

# Hide all geometry so the camera sees only the world.
for obj in bpy.data.objects:
    if obj.type in ("MESH", "CURVE", "SURFACE", "FONT", "META"):
        obj.hide_render = True

# Switch the active camera to a panoramic equirect one.
cam = bpy.data.cameras.new("equirect_cam")
cam.type = "PANO"
# Panoramic type lives on the cycles substruct in Blender 5.x
try:
    cam.cycles.panorama_type = "EQUIRECTANGULAR"
except Exception:
    cam.panorama_type = "EQUIRECTANGULAR"
cam_obj = bpy.data.objects.new("equirect_cam_obj", cam)
scene.collection.objects.link(cam_obj)
# vibrt's grid layout uses theta from +Z (top), phi=atan2(y,x) where
# the center column (u=0.5) points to -X. Cycles' equirect with an
# identity camera puts +Y at top and -Z at center — different. Build
# a camera matrix whose columns are (cam_right, cam_up, cam_back) =
# (-Y_w, +Z_w, +X_w) so that pixel (0.5, 0.5) maps to camera forward
# = -X_w and pixel (·, 0) maps to camera up = +Z_w. That makes the
# Cycles equirect directly pixel-comparable with our host grid.
import mathutils
cam_obj.matrix_world = mathutils.Matrix((
    ( 0.0, 0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0, 0.0),
    ( 0.0, 1.0, 0.0, 0.0),
    ( 0.0, 0.0, 0.0, 1.0),
))
scene.camera = cam_obj

scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.render.resolution_x = 1024
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "32"
scene.view_settings.view_transform = "Raw"

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(HERE, "test_scenes", "pabellon", "_cycles_equirect_world.exr")
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")
