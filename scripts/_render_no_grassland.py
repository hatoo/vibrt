"""Hide both Plane (cavity rim) and tree_scatter_plane (the 713-vertex
grass terrain). If the slope wedge is still visible, look elsewhere.
If it disappears, tree_scatter_plane is the offender."""
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

for n in ("Plane", "tree_scatter_plane"):
    obj = bpy.data.objects.get(n)
    if obj:
        obj.hide_render = True
        print(f"hid {n}")

scene.render.filepath = os.path.join(HERE, "test_scenes", "pabellon",
                                     "_no_grassland.png")
bpy.ops.render.render(write_still=True)
print(f"wrote {scene.render.filepath}")
