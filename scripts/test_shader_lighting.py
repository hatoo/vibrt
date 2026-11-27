"""Compare procedural mesh-light illumination with Cycles; --layered adds coat/sheen."""
from pathlib import Path
import json
import sys

import bpy
from mathutils import Vector
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
import vibrt_blender
from vibrt_blender import exporter, vibrt_native as native

bpy.ops.wm.read_factory_settings(use_empty=True)
vibrt_blender.register()
scene = bpy.context.scene
scene.render.resolution_x = scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 512
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.cycles.max_bounces = 2
scene.cycles.pixel_filter_type = "BOX"
scene.render.filter_size = 0.01
world = bpy.data.worlds.new("Black")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
scene.world = world

bpy.ops.mesh.primitive_plane_add(size=12)
floor = bpy.context.object
mat = bpy.data.materials.new("Diffuse receiver")
mat.use_nodes = True
tree = mat.node_tree
tree.nodes.clear()
diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
diffuse.inputs["Color"].default_value = (0.7, 0.7, 0.7, 1)
out = tree.nodes.new("ShaderNodeOutputMaterial")
tree.links.new(diffuse.outputs[0], out.inputs["Surface"])
floor.data.materials.append(mat)

bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 2))
light = bpy.context.object
light.visible_camera = False
mat = bpy.data.materials.new("Procedural emitter")
mat.use_nodes = True
tree = mat.node_tree
tree.nodes.clear()
checker = tree.nodes.new("ShaderNodeTexChecker")
checker.inputs["Color1"].default_value = (1, 0.1, 0.1, 1)
checker.inputs["Color2"].default_value = (0.1, 0.1, 1, 1)
checker.inputs["Scale"].default_value = 4.5
layered = "--layered" in sys.argv
emission = tree.nodes.new("ShaderNodeBsdfPrincipled" if layered else "ShaderNodeEmission")
if layered:
    emission.inputs["Base Color"].default_value = (0, 0, 0, 1)
    emission.inputs["Coat Weight"].default_value = 0.8
    emission.inputs["Coat Tint"].default_value = (0.2, 0.7, 0.9, 1)
    emission.inputs["Sheen Weight"].default_value = 0.6
    emission.inputs["Sheen Tint"].default_value = (0.8, 0.3, 0.5, 1)
emission.inputs["Emission Strength" if layered else "Strength"].default_value = 20
out = tree.nodes.new("ShaderNodeOutputMaterial")
tree.links.new(checker.outputs["Color"], emission.inputs["Emission Color" if layered else "Color"])
tree.links.new(emission.outputs[0], out.inputs["Surface"])
light.data.materials.append(mat)

many = "--many" in sys.argv
if many:
    # Distinct groups exercise every tree branch and the matching hit-side PMF.
    for x, y in ((-3, -3), (0, -3), (3, -3), (-3, 0), (3, 0), (-3, 3), (0, 3), (3, 3)):
        other = light.copy()
        scene.collection.objects.link(other)
        other.location.x += x
        other.location.y += y

camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
scene.collection.objects.link(camera)
scene.camera = camera
camera.location = (4, -6, 4)
camera.rotation_euler = (Vector((0, 0, 0)) - camera.location).to_track_quat("-Z", "Y").to_euler()

directory = ROOT / ("target/shader-lighting-many" if many else "target/shader-lighting-layered" if layered else "target/shader-lighting")
directory.mkdir(parents=True, exist_ok=True)
scene.render.filepath = str(directory / "cycles.exr")
bpy.ops.render.render(write_still=True)
image = bpy.data.images.load(scene.render.filepath, check_existing=False)
reference = np.empty(64 * 64 * 4, np.float32)
image.pixels.foreach_get(reference)
reference = reference.reshape(64, 64, 4)
scene.render.engine = "VIBRT"
scene.vibrt_spp = 512
exported, blobs, textures = exporter.export_scene_to_memory(bpy.context.evaluated_depsgraph_get(), serialize=False)
assert len(exported["mesh_lights"]) == (9 if many else 1)
assert any("emission" in m.get("shader_graph", {}).get("outputs", {}) for m in exported["materials"])
actual, _ = native.render(exported, [b.view(np.uint8) for b in blobs],
                          {"spp":512, "width":64, "height":64}, texture_arrays=textures)
np.save(directory / "cuda.npy", actual)
mask = np.zeros((64, 64), bool)
mask[12:52, 12:52] = True
mask &= np.max(reference[:, :, :3], axis=-1) > 0.01
assert np.count_nonzero(mask) > 800
expected_mean = reference[:, :, :3][mask].mean(axis=0)
actual_mean = actual[:, :, :3][mask].mean(axis=0)
relative_error = np.abs(expected_mean - actual_mean) / expected_mean
result = {"cycles_mean":expected_mean.tolist(), "cuda_mean":actual_mean.tolist(),
          "relative_error":relative_error.tolist()}
(directory / "results.json").write_text(json.dumps(result, indent=2))
print("[shader-lighting]", result, flush=True)
assert np.isfinite(actual).all()
assert np.max(relative_error) < 0.07, "Procedural emission must illuminate other surfaces"
