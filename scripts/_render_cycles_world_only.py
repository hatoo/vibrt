"""Render the actual scene with Cycles but with all geometry hidden, so
the camera sees only the world background. This gives the ground-truth
'what Cycles thinks the sky looks like from the camera's perspective'
that we can compare to vibrt's bake-then-sample output."""
import bpy
import os, sys

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_depth = "8"

# Hide all visible geometry so only the world surface is captured.
for obj in bpy.data.objects:
    if obj.type in ("MESH", "CURVE", "SURFACE", "FONT", "META"):
        obj.hide_render = True

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(HERE, "test_scenes", "pabellon", "_cycles_world_only.png")
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")
