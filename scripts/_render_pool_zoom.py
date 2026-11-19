"""Render pabellon at the same camera but DOUBLED resolution so we can
zoom in on the pool floor and see what's there."""
import bpy, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

scene = bpy.context.scene
scene.render.engine = "VIBRT"
scene.vibrt_spp = 256
scene.render.resolution_x = 2560
scene.render.resolution_y = 1440
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_depth = "8"

out = os.path.join(HERE, "test_scenes", "pabellon", "_zoom_vibrt.png")
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")
