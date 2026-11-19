"""Render pabellon in Cycles with the Sun lamp hidden, so we can
quantify the envmap-only contribution and see whether vibrt's
envmap-only rendering matches in colour balance."""
import bpy, os
scene = bpy.context.scene
for o in bpy.data.objects:
    if o.type == "LIGHT":
        o.hide_render = True

scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.cycles.use_denoising = True
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 50
scene.render.image_settings.file_format = "PNG"

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scene.render.filepath = os.path.join(HERE, "test_scenes", "pabellon",
                                      "_cycles_envmap_only.png")
bpy.ops.render.render(write_still=True)
print(f"wrote {scene.render.filepath}")
