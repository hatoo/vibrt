"""Render pabellon to EXR (linear-light HDR) in both Cycles and vibrt
so we can compare the actual radiance values without any view_transform
between us and the truth. If the EXR pixels match, the visual
difference was tone-map-related; if they differ, the linear renderer
itself produces different lighting.
"""
import bpy, os, sys

scene = bpy.context.scene
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.view_settings.view_transform = "Raw"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 50

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mode = os.environ.get("VIBRT_TEST_MODE", "vibrt")
if mode == "cycles":
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    out = os.path.join(HERE, "test_scenes", "pabellon", "_pabellon_cycles_lin.exr")
else:
    sys.path.insert(0, os.path.join(HERE, "blender"))
    import vibrt_blender  # noqa
    try: bpy.ops.preferences.addon_enable(module="vibrt_blender")
    except Exception: pass
    scene.render.engine = "VIBRT"
    try: scene.vibrt_spp = 64
    except Exception: pass
    out = os.path.join(HERE, "test_scenes", "pabellon", "_pabellon_vibrt_lin.exr")

scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")
