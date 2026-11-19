"""Render pabellon's Sky Texture in isolation (no MixShader, no HDRI)
via Cycles equirect, so we can see whether PREETHAM at sun_elevation
=90° actually produces a visible hot sun disc when rendered the same
way our prebake does."""
import bpy, os, math, mathutils

scene = bpy.context.scene

# Find the Sky Texture node and clone its params into a fresh world.
src_world = scene.world
sky_node = None
for n in src_world.node_tree.nodes:
    if n.bl_idname == "ShaderNodeTexSky":
        sky_node = n
        break
print(f"sky_type={sky_node.sky_type} sun_disc={sky_node.sun_disc} "
      f"sun_size={sky_node.sun_size} sun_elevation={math.degrees(sky_node.sun_elevation):.1f}°")

w = bpy.data.worlds.new("__sky_only")
w.use_nodes = True
nt = w.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputWorld")
bg = nt.nodes.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = 1.0  # bake at unit strength
sky = nt.nodes.new("ShaderNodeTexSky")
for attr in ("sky_type", "sun_disc", "sun_size", "sun_intensity",
             "sun_elevation", "sun_rotation", "altitude",
             "air_density", "ozone_density", "ground_albedo", "turbidity"):
    if hasattr(sky_node, attr) and hasattr(sky, attr):
        try: setattr(sky, attr, getattr(sky_node, attr))
        except Exception: pass
nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
scene.world = w

for o in bpy.data.objects:
    if o.type in ("MESH","CURVE","SURFACE","FONT","META","LIGHT"):
        o.hide_render = True

cam = bpy.data.cameras.new("eq_cam")
cam.type = "PANO"
try: cam.cycles.panorama_type = "EQUIRECTANGULAR"
except Exception: cam.panorama_type = "EQUIRECTANGULAR"
cam_obj = bpy.data.objects.new("eq_cam_obj", cam)
scene.collection.objects.link(cam_obj)
# Same camera matrix the prebake uses (rotation_euler=(-π/2, 0, 0))
cam_obj.matrix_world = mathutils.Euler((-math.pi/2, 0, 0)).to_matrix().to_4x4()
scene.camera = cam_obj

scene.render.engine = "CYCLES"
scene.cycles.samples = 256
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.render.resolution_x = 1024
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "32"
scene.view_settings.view_transform = "Raw"

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(HERE, "test_scenes", "pabellon", "_cycles_sky_only.exr")
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")

# Sample top center
import numpy as np
img = bpy.data.images.load(out, check_existing=False)
img.colorspace_settings.name = "Non-Color"
arr = np.empty(img.size[0]*img.size[1]*4, dtype=np.float32)
img.pixels.foreach_get(arr)
rgba = arr.reshape((img.size[1], img.size[0], 4))[::-1]
rgb = rgba[..., :3]
print(f"shape={rgb.shape}  mean={tuple(round(float(x),3) for x in rgb.mean(axis=(0,1)))}  max={float(rgb.max()):.2f}  median={float(np.median(rgb)):.4f}")
print(f"top-row mean: {tuple(round(float(x),3) for x in rgb[0].mean(axis=0))}  max in top-row: {float(rgb[0].max()):.2f}")
print(f"top 1% pixels: count={int((rgb.mean(axis=2) > np.percentile(rgb.mean(axis=2), 99)).sum())}")
yy, xx = np.unravel_index(np.argmax(rgb.mean(axis=2)), rgb.shape[:2])
print(f"brightest at ({xx},{yy}) uv=({xx/rgb.shape[1]:.3f}, {yy/rgb.shape[0]:.3f}) value={tuple(round(float(c),2) for c in rgb[yy,xx])}")
