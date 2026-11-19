"""Tone-map preview_skybake.exr to PNG using the scene's view settings
so we can eyeball it next to the Cycles preview.

Run via: blender -b empty.blend --python scripts/_dump_skybake_png.py
   or:    blender -b <any.blend> --python scripts/_dump_skybake_png.py
"""

import os
import bpy

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exr = os.path.join(HERE, "test_scenes", "lone_monk", "preview_skybake.exr")
out_filmic = os.path.join(HERE, "test_scenes", "lone_monk", "preview_skybake_filmic.png")
out_raw = os.path.join(HERE, "test_scenes", "lone_monk", "preview_skybake_raw.png")

img = bpy.data.images.load(exr)
img.colorspace_settings.name = "Linear Rec.709"

# Tone-map with Filmic + exposure -2 (matching lone_monk)
sc = bpy.context.scene
sc.view_settings.view_transform = "Filmic"
sc.view_settings.look = "Medium Contrast"
sc.view_settings.exposure = -2.0
sc.view_settings.gamma = 1.0
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGB"
sc.render.image_settings.color_depth = "8"

img.save_render(filepath=out_filmic)
print(f"wrote {out_filmic}")

sc.view_settings.view_transform = "Raw"
sc.view_settings.exposure = 0.0
img.save_render(filepath=out_raw)
print(f"wrote {out_raw}")
