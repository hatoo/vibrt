"""Render pabellon via Cycles with water hidden — apples-to-apples
comparison with vibrt's `_no_water.png`."""
import bpy, os

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

for name in ("water_plane_still", "lotus_scattering_plane"):
    obj = bpy.data.objects.get(name)
    if obj:
        obj.hide_render = True

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(HERE, "test_scenes", "pabellon", "_cycles_no_water.png")
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")
