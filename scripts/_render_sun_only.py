"""Render pabellon lit ONLY by the Sun lamp (Lamp.001) — replace the
world graph with a constant black background so envmap NEE / BSDF-miss
return zero. Whatever brightness remains is purely from the sun light's
direct + indirect contribution.

If the building's front / roof read as warm orange here, the sun light
itself works and the Cycles-vibrt highlight gap is in envmap territory.
If they're black, sun NEE isn't reaching them.
"""
import bpy, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

scene = bpy.context.scene

# Black world.
w = bpy.data.worlds.new("__black")
w.use_nodes = True
nt = w.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputWorld")
bg = nt.nodes.new("ShaderNodeBackground")
bg.inputs["Color"].default_value = (0, 0, 0, 1)
bg.inputs["Strength"].default_value = 0.0
nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
scene.world = w

scene.render.engine = "VIBRT"
try:
    scene.vibrt_spp = 64
except Exception:
    pass
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 50
scene.render.image_settings.file_format = "PNG"

scene.render.filepath = os.path.join(HERE, "test_scenes", "pabellon",
                                     "_sun_only.png")
bpy.ops.render.render(write_still=True)
print(f"wrote {scene.render.filepath}")
