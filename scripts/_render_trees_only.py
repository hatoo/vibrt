"""Render with everything but trees + terrain hidden so we can see
whether trees actually appear in the camera's view. If trees are
properly visible, they should mask the curved terrain — matching
Cycles. If they don't, alpha cutout / scale / placement is broken."""
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

# Keep camera + tree_scatter_plane (terrain emitter) + tree_scatter (the
# instanced source object). Hide everything else.
keep_names = {"tree_scatter_plane", "tree_scatter"}
keep_collections = {"tree"}
for obj in bpy.data.objects:
    if obj.type in ("MESH", "CURVE", "SURFACE", "FONT", "META", "EMPTY"):
        obj.hide_render = obj.name not in keep_names
    # Children of the 'tree' collection used as instances need to remain
    # available (depsgraph still pulls them via the particle settings).

scene.render.filepath = os.path.join(HERE, "test_scenes", "pabellon",
                                     "_trees_only.png")
bpy.ops.render.render(write_still=True)
print(f"wrote {scene.render.filepath}")
