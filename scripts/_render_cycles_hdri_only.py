"""Bake the HDRI side of pabellon's Mixed world in isolation: replace
the world graph with `Output ← Background ← TexImage(DSC_8129) ←
Mapping(orig rot) ← TexCoord.Object`. The result is the clean ground
truth for Cycles' (R, projection, extension, sd.P) interpretation of
this layer."""
import bpy, os, sys, mathutils

scene = bpy.context.scene
world = scene.world

# Save original mapping rotation by reading the original world.
src_image = bpy.data.images["DSC_8129.JPG"]
mapping_rot = None
for n in world.node_tree.nodes:
    if n.bl_idname == "ShaderNodeMapping":
        rx, ry, rz = n.inputs["Rotation"].default_value
        mapping_rot = (float(rx), float(ry), float(rz))
        print(f"original mapping rot = {mapping_rot}")

# Build a fresh tiny world that uses ONLY the HDRI side.
new_world = bpy.data.worlds.new("__hdri_only")
new_world.use_nodes = True
nt = new_world.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputWorld")
bg = nt.nodes.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = 1.0
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = src_image
tex.projection = "FLAT"
tex.extension = "REPEAT"
mp = nt.nodes.new("ShaderNodeMapping")
mp.vector_type = "POINT"
import os as _os
if _os.environ.get("VIBRT_NO_ROT") == "1":
    print("identity rotation override")
    mp.inputs["Rotation"].default_value = (0.0, 0.0, 0.0)
else:
    mp.inputs["Rotation"].default_value = mapping_rot
tc = nt.nodes.new("ShaderNodeTexCoord")
tc.object = None
nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
nt.links.new(tex.outputs["Color"], bg.inputs["Color"])
nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
scene.world = new_world

# Hide all geometry.
for obj in bpy.data.objects:
    if obj.type in ("MESH", "CURVE", "SURFACE", "FONT", "META"):
        obj.hide_render = True

# Equirect camera aligned to vibrt convention.
cam = bpy.data.cameras.new("eq")
cam.type = "PANO"
try: cam.cycles.panorama_type = "EQUIRECTANGULAR"
except Exception: cam.panorama_type = "EQUIRECTANGULAR"
cam_obj = bpy.data.objects.new("eq_obj", cam)
scene.collection.objects.link(cam_obj)
cam_obj.matrix_world = mathutils.Matrix((
    ( 0.0, 0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0, 0.0),
    ( 0.0, 1.0, 0.0, 0.0),
    ( 0.0, 0.0, 0.0, 1.0),
))
scene.camera = cam_obj

scene.render.engine = "CYCLES"
scene.cycles.samples = 16
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.render.resolution_x = 1024
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
suffix = "_norot" if os.environ.get("VIBRT_NO_ROT") == "1" else ""
out_path = os.path.join(HERE, "test_scenes", "pabellon",
                         f"_cycles_hdri_only_eq{suffix}.png")
scene.render.filepath = out_path
bpy.ops.render.render(write_still=True)
print(f"wrote {out_path}")
