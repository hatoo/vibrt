"""Render pabellon with water enabled but lotus pads hidden, to isolate
underwater visibility from any lotus-pad shadowing."""
import bpy, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

# Hide lotus pads (and their scatterer) so we can see the water surface alone.
for name in ("lotus_scattering_plane",):
    obj = bpy.data.objects.get(name)
    if obj:
        obj.hide_render = True

scene = bpy.context.scene
scene.render.engine = "VIBRT"
scene.vibrt_spp = 256
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

out = os.path.join(HERE, "test_scenes", "pabellon", "_no_lotus.png")
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")
