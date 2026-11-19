"""Render pabellon with the suspected terrain `Plane` (the 150-vertex
cavity mesh) hidden. If the bottom-right 'weird slope' disappears, that
mesh is the offender."""
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
    scene.vibrt_spp = 64
except Exception:
    pass
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 50
scene.render.image_settings.file_format = "PNG"

obj = bpy.data.objects.get("Plane")
if obj:
    obj.hide_render = True
    print(f"hid {obj.name}")

scene.render.filepath = os.path.join(HERE, "test_scenes", "pabellon",
                                     "_no_terrain.png")
bpy.ops.render.render(write_still=True)
print(f"wrote {scene.render.filepath}")
