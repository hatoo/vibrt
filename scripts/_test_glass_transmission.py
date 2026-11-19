"""Build a minimal scene programmatically: a glass plane over a red diffuse
plane, lit by a sun. Render via vibrt and report the centre-pixel color. If
the glass transmits properly we should see red."""
import bpy, os, sys, math

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

# Wipe scene.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat, do_unlink=True)
for me in list(bpy.data.meshes):
    bpy.data.meshes.remove(me, do_unlink=True)

scene = bpy.context.scene
scene.world = bpy.data.worlds.new("simple")
scene.world.use_nodes = True
nt = scene.world.node_tree
bg = nt.nodes["Background"]
bg.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
bg.inputs["Strength"].default_value = 0.5

# Red plane below.
red_mat = bpy.data.materials.new("red")
red_mat.use_nodes = True
red_mat.node_tree.nodes.clear()
out_n = red_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
diff = red_mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
diff.inputs["Color"].default_value = (1.0, 0.0, 0.0, 1.0)
red_mat.node_tree.links.new(diff.outputs[0], out_n.inputs["Surface"])

bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
red_plane = bpy.context.object
red_plane.name = "red_plane"
red_plane.data.materials.append(red_mat)

# Glass plane on top.
glass_mat = bpy.data.materials.new("glass")
glass_mat.use_nodes = True
glass_mat.node_tree.nodes.clear()
out_g = glass_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
gb = glass_mat.node_tree.nodes.new("ShaderNodeBsdfGlass")
gb.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
gb.inputs["Roughness"].default_value = 0.0548
gb.inputs["IOR"].default_value = 1.1
glass_mat.node_tree.links.new(gb.outputs[0], out_g.inputs["Surface"])

bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 1.0))
glass_plane = bpy.context.object
glass_plane.name = "glass_plane"
glass_plane.data.materials.append(glass_mat)

# Camera at moderate grazing — z=1.7 (0.7m above glass at z=1.0), looking
# along -Y at 80° from world up (so pretty grazing).
bpy.ops.object.camera_add(location=(0, -10, 1.7), rotation=(math.radians(85), 0, 0))
scene.camera = bpy.context.object

# Sun
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
sun = bpy.context.object
sun.data.energy = 5.0

scene.render.engine = "VIBRT"
scene.vibrt_spp = 64
scene.render.resolution_x = 600
scene.render.resolution_y = 400
scene.render.image_settings.file_format = "PNG"

out = os.path.join(HERE, "test_scenes", "_test_glass.png")
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"wrote {out}")
