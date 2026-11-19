"""Render with the water set to perfect smooth IOR=1.0 (no refraction
bending, no Fresnel reflection). If the pool floor becomes clearly
visible, the issue is in our Fresnel/refraction code; if it's still
opaque, something else is wrong."""
import bpy, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

mat = bpy.data.materials.get("water")
gb = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfGlass")
gb.inputs["IOR"].default_value = 1.0  # no Fresnel, no refraction bending
gb.inputs["Roughness"].default_value = 0.0
sock = gb.inputs.get("Normal")
if sock and sock.is_linked:
    for link in list(sock.links):
        mat.node_tree.links.remove(link)

scene = bpy.context.scene
scene.render.engine = "VIBRT"
scene.vibrt_spp = 128
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

out = os.path.join(HERE, "test_scenes", "pabellon", "_water_clear.png")
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")
