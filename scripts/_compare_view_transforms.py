"""Re-encode both vibrt and Cycles linear EXRs through several
Blender view transforms (Standard, Filmic, AgX) and save to PNG to
see how each transform changes the visual appearance and whether any
of them reduces the perceptual gap."""
import bpy, os, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scene = bpy.context.scene
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "8"
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 50

# Use the EXR-as-input compositor pipeline
# Easier: load EXR, set scene properties for view_transform, save as PNG

for src_name in ("vibrt", "cycles"):
    src = os.path.join(HERE, "test_scenes", "pabellon", f"_pabellon_{src_name}_lin.exr")
    img = bpy.data.images.load(src, check_existing=False)
    img.colorspace_settings.name = "Non-Color"
    iw, ih = img.size

    for vt in ("Standard", "Filmic", "AgX Base sRGB"):
        try:
            scene.view_settings.view_transform = vt
        except Exception:
            print(f"  skip {vt}")
            continue
        out = os.path.join(HERE, "test_scenes", "pabellon",
                            f"_tone_{src_name}_{vt.replace(' ', '_')}.png")
        img.save_render(filepath=out, scene=scene)
        print(f"  wrote {out}")
