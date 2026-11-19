"""Render pabellon lit ONLY by the Mixed envmap (no Sun lamp) — keep
the original world but hide Lamp.001 so we measure how much brightness
the rasterised mixed grid + the residual Sky bake (sun included if not
extracted) contribute on their own."""
import bpy, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

scene = bpy.context.scene
for o in bpy.data.objects:
    if o.type == "LIGHT":
        o.hide_render = True
        print(f"hid light {o.name}")

scene.render.engine = "VIBRT"
try:
    scene.vibrt_spp = 64
except Exception:
    pass
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 50
scene.render.image_settings.file_format = "PNG"

scene.render.filepath = os.path.join(HERE, "test_scenes", "pabellon",
                                     "_envmap_only.png")
bpy.ops.render.render(write_still=True)
print(f"wrote {scene.render.filepath}")
