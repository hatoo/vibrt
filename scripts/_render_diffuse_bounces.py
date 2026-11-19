"""Render pabellon at varying diffuse_bounces to localise where the
over-brightness vs cycles comes from.

  bounces=0 → pure NEE on the camera ray's first surface (no indirect)
  bounces=1 → adds one diffuse bounce
  bounces=3 → pabellon default

If basin brightness scales linearly with bounce count we have an
honest accumulation difference; if it's already wrong at bounces=0
the bug is in NEE / first-surface contribution."""
import bpy, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

# Hide water + lotus so the basin pixels we sample are pure basin.
for name in ("water_plane_still", "lotus_scattering_plane"):
    obj = bpy.data.objects.get(name)
    if obj:
        obj.hide_render = True

scene = bpy.context.scene
scene.render.engine = "VIBRT"
scene.vibrt_spp = 256  # tighter MC noise so we can read brightness reliably
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

import numpy as np
def basin_stats(path):
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "sRGB"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32); img.pixels.foreach_get(flat)
    arr = flat.reshape((h, w, 4))[..., :3]
    box = arr[h-1-650:h-1-580, 200:1080, :]
    return (tuple(round(float(c),3) for c in box.mean(axis=(0,1))),
            tuple(round(float(c),3) for c in box.std(axis=(0,1))))

for db in (0, 1, 3):
    scene.cycles.diffuse_bounces = db
    out = os.path.join(HERE, "test_scenes", "pabellon",
                       f"_basin_diff{db}.png")
    scene.render.filepath = out
    bpy.ops.render.render(write_still=True)
    mean, std = basin_stats(out)
    print(f"diffuse_bounces={db}: basin mean={mean} std={std}")
