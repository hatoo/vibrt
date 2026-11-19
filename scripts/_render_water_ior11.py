"""Try IOR=1.1 with no roughness or bump — pure smooth glass with the
authored IOR. If the water is still opaque, the issue is in the Fresnel
/ transmission lobe."""
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
gb.inputs["IOR"].default_value = 1.1
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

out = os.path.join(HERE, "test_scenes", "pabellon", "_water_ior11.png")
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")
