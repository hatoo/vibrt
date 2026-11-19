"""Drop a Plane.003-shaped quad with `pebbles.ground` (rocks.jpg) into a
clean uniform-envmap scene and render. Expected: a tiled rocky pattern at
the same UV scale Cycles renders. If vibrt's tile rate / contrast match
expectations the basin issue in pabellon is purely a brightness scaling
matter, not a UV / texture sampling bug."""
import bpy, os, sys, math
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

# Wipe everything except the pebbles.ground material we want to reuse.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

scene = bpy.context.scene
# Constant grey envmap so we know exactly how much light hits the plane.
scene.world = bpy.data.worlds.new("flat_grey")
scene.world.use_nodes = True
nt = scene.world.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputWorld")
bg = nt.nodes.new("ShaderNodeBackground")
bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
bg.inputs["Strength"].default_value = 1.0
nt.links.new(bg.outputs[0], out.inputs["Surface"])

# Plane sized 20×12 m matching the pool basin's UV span.
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
plane = bpy.context.object
plane.name = "basin_test"
plane.scale = (10.0, 6.0, 1.0)
bpy.ops.object.transform_apply(scale=True)
mat = bpy.data.materials.get("pebbles.ground")
if mat is None:
    raise RuntimeError("pebbles.ground material not present")
plane.data.materials.append(mat)
# Force the same UV range as Plane.003 (UV span [-9..10, -4..7]).
me = plane.data
uv_layer = me.uv_layers.active
if uv_layer is None:
    uv_layer = me.uv_layers.new(name="UVMap")
# Map the four corners to the expected UV coords.
corners_world = [(plane.matrix_world @ v.co) for v in me.vertices]
# Sort by xy so we can map deterministically. Plane corners are at (-10,-6), (10,-6), (10,6), (-10,6).
# Map UV (-9..10) and (-4..7).
def uv_for(co):
    u = -9.0 + (co.x + 10.0) * (10.0 - (-9.0)) / 20.0
    v = -4.0 + (co.y + 6.0) * (7.0 - (-4.0)) / 12.0
    return (u, v)
for poly in me.polygons:
    for li in poly.loop_indices:
        v_idx = me.loops[li].vertex_index
        co = me.vertices[v_idx].co
        wpos = plane.matrix_world @ co
        uv_layer.data[li].uv = uv_for(wpos)

# Camera looking down at the plane.
bpy.ops.object.camera_add(location=(0, 0, 12), rotation=(0, 0, 0))
scene.camera = bpy.context.object

scene.render.engine = "VIBRT"
scene.vibrt_spp = 64
scene.render.resolution_x = 800
scene.render.resolution_y = 480
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"

out_path = os.path.join(HERE, "test_scenes", "_basin_uv_test.png")
scene.render.filepath = out_path
bpy.ops.render.render(write_still=True)
print(f"wrote {out_path}")

# Sample a few pixels to characterise the result.
import numpy as np
img = bpy.data.images.load(out_path)
img.colorspace_settings.name = "sRGB"
w, h = int(img.size[0]), int(img.size[1])
flat = np.empty(w * h * 4, dtype=np.float32); img.pixels.foreach_get(flat)
arr = flat.reshape((h, w, 4))[..., :3]
# centre 200×200 box
cy = h // 2; cx = w // 2
box = arr[cy - 100:cy + 100, cx - 100:cx + 100, :]
print(f"basin patch mean rgb = {tuple(round(float(x),3) for x in box.mean(axis=(0,1)))}")
print(f"basin patch std rgb  = {tuple(round(float(x),3) for x in box.std(axis=(0,1)))}")
print(f"basin patch p10/p50/p90 (R) = "
      f"{tuple(round(float(np.percentile(box[..., 0], p)),3) for p in (10, 50, 90))}")
