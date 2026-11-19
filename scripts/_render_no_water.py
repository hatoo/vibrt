"""Render pabellon with the water plane hidden so we can see whether
pebbles below are being lit and rendered correctly. If the pool floor
shows distinct pebbles, the issue is purely in transmission. If the
floor is also dark, the pebble material itself isn't being lit."""
import bpy, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

# Hide the water plane and the lotus scatter plane (the pads above the water).
for name in ("water_plane_still", "lotus_scattering_plane"):
    obj = bpy.data.objects.get(name)
    if obj:
        obj.hide_render = True

scene = bpy.context.scene
scene.render.engine = "VIBRT"
scene.vibrt_spp = 64
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

out = os.path.join(HERE, "test_scenes", "pabellon", "_no_water.png")
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")
