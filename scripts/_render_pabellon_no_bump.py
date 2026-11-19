"""Render pabellon with the water material's bump disconnected to test
whether the bump is what's breaking pebble visibility through the water."""
import bpy, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

# Disconnect bump from the Glass BSDF.
mat = bpy.data.materials.get("water")
gb = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfGlass")
sock = gb.inputs.get("Normal")
if sock and sock.is_linked:
    for link in list(sock.links):
        mat.node_tree.links.remove(link)
    print("disconnected water bump")

scene = bpy.context.scene
scene.render.engine = "VIBRT"
scene.vibrt_spp = 128
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

out = os.path.join(HERE, "test_scenes", "pabellon", "_no_bump.png")
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")
