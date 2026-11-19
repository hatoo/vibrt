"""Override clamps very low (0.3) to verify the clamp is taking effect."""
import bpy, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

scene = bpy.context.scene
scene.cycles.sample_clamp_direct = 0.3
scene.cycles.sample_clamp_indirect = 0.3
scene.render.engine = "VIBRT"
scene.vibrt_spp = 128
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

out = os.path.join(HERE, "test_scenes", "pabellon", "_clamp_low.png")
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")
