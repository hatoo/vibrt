"""Render the actual scene with vibrt (forced engine=VIBRT) with all
geometry hidden so only the world background is visible. Lets us compare
the sky plate apples-to-apples with `_cycles_world_only.py`."""
import bpy, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa: F401

try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

scene = bpy.context.scene
scene.render.engine = "VIBRT"
try:
    scene.vibrt_spp = 16
except Exception:
    pass
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_depth = "8"
scene.view_settings.view_transform = "Standard"

for obj in bpy.data.objects:
    if obj.type in ("MESH", "CURVE", "SURFACE", "FONT", "META"):
        obj.hide_render = True

out = os.path.join(HERE, "test_scenes", "pabellon", "_vibrt_world_only.png")
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")
