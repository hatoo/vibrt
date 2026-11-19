"""Render PREETHAM Sky alone in a *fresh* scene (so no clamp_direct
inheritance), at high res + high spp + EXR float, with view_transform
forced Raw. Verify whether the sun disc actually has HDR luminance.
"""
import bpy, os, math, mathutils, numpy as np

src_world = bpy.context.scene.world
sky_node = next((n for n in src_world.node_tree.nodes
                  if n.bl_idname == "ShaderNodeTexSky"), None)

# Brand new scene
tmp = bpy.data.scenes.new("__pristine_sky")
print(f"new scene clamp_direct = {tmp.cycles.sample_clamp_direct}")
print(f"new scene clamp_indirect = {tmp.cycles.sample_clamp_indirect}")

# Fresh world
w = bpy.data.worlds.new("__sky_pristine")
w.use_nodes = True
nt = w.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputWorld")
bg = nt.nodes.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = 1.0
sky = nt.nodes.new("ShaderNodeTexSky")
for attr in ("sky_type", "sun_disc", "sun_size", "sun_intensity",
             "sun_elevation", "sun_rotation", "altitude",
             "air_density", "ozone_density", "ground_albedo", "turbidity"):
    if hasattr(sky_node, attr) and hasattr(sky, attr):
        try: setattr(sky, attr, getattr(sky_node, attr))
        except Exception: pass
nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
tmp.world = w

cam_data = bpy.data.cameras.new("eq")
cam_data.type = "PANO"
try: cam_data.cycles.panorama_type = "EQUIRECTANGULAR"
except Exception: cam_data.panorama_type = "EQUIRECTANGULAR"
cam_obj = bpy.data.objects.new("eq_obj", cam_data)
tmp.collection.objects.link(cam_obj)
cam_obj.matrix_world = mathutils.Euler((-math.pi/2, 0, 0)).to_matrix().to_4x4()
tmp.camera = cam_obj

tmp.render.engine = "CYCLES"
tmp.cycles.samples = 1024
tmp.cycles.use_denoising = False
tmp.cycles.use_adaptive_sampling = False
tmp.render.resolution_x = 2048
tmp.render.resolution_y = 1024
tmp.render.resolution_percentage = 100
tmp.render.image_settings.file_format = "OPEN_EXR"
tmp.render.image_settings.color_mode = "RGB"
tmp.render.image_settings.color_depth = "32"
tmp.view_settings.view_transform = "Raw"

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exr = os.path.join(HERE, "test_scenes", "pabellon", "_pristine_sky.exr")
tmp.render.filepath = exr
bpy.ops.render.render(scene=tmp.name, write_still=True)
print(f"wrote {exr}")

img = bpy.data.images.load(exr, check_existing=False)
img.colorspace_settings.name = "Non-Color"
arr = np.empty(img.size[0]*img.size[1]*4, dtype=np.float32)
img.pixels.foreach_get(arr)
rgba = arr.reshape((img.size[1], img.size[0], 4))[::-1]
rgb = rgba[..., :3]
print(f"shape={rgb.shape} mean={tuple(round(float(x),3) for x in rgb.mean(axis=(0,1)))} max={float(rgb.max()):.2f}")
lum = 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]
print(f"luminance max={float(lum.max()):.2f} p50={float(np.percentile(lum, 50)):.4f} p99={float(np.percentile(lum, 99)):.4f}")
print(f"max/median ratio = {float(lum.max()/max(np.median(lum), 1e-4)):.2f}")
yy, xx = np.unravel_index(np.argmax(lum), lum.shape)
print(f"brightest at ({xx},{yy}) uv=({xx/rgb.shape[1]:.3f}, {yy/rgb.shape[0]:.3f}) value={tuple(round(float(c),2) for c in rgb[yy,xx])}")
