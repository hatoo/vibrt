"""Headless Blender integration: UI operator, render-mode snapshot, cache reuse."""

from pathlib import Path
import json
import sys
from unittest.mock import patch

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
import vibrt_blender
from vibrt_blender import exporter, shader_precompile, vibrt_native as native

directory = ROOT / "target/shader-precompile"
directory.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
vibrt_blender.register()
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.render.resolution_x = scene.render.resolution_y = 32
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
bpy.ops.mesh.primitive_plane_add(size=2)
mat = bpy.data.materials.new("Unsaved shader edits")
mat.use_nodes = True
bpy.context.object.data.materials.append(mat)
tree = mat.node_tree
tree.nodes.clear()
output = tree.nodes.new("ShaderNodeOutputMaterial")
toon = tree.nodes.new("ShaderNodeBsdfToon")
noise = tree.nodes.new("ShaderNodeTexNoise")
tree.links.new(noise.outputs["Color"], toon.inputs["Color"])
tree.links.new(toon.outputs[0], output.inputs["Surface"])
aov = bpy.context.view_layer.aovs.add()
aov.name = "Noise"
aov.type = "VALUE"
node = tree.nodes.new("ShaderNodeOutputAOV")
node.aov_name = aov.name
tree.links.new(noise.outputs["Fac"], node.inputs["Value"])
# A constant Math chain must disappear before both precompile and rendering.
folded_aov = bpy.context.view_layer.aovs.add()
folded_aov.name = "Folded"
folded_aov.type = "VALUE"
folded_output = tree.nodes.new("ShaderNodeOutputAOV")
folded_output.aov_name = folded_aov.name
multiply = tree.nodes.new("ShaderNodeMath")
multiply.operation = "MULTIPLY"
multiply.inputs[0].default_value = 2.0
multiply.inputs[1].default_value = 3.0
tree.links.new(multiply.outputs[0], folded_output.inputs["Value"])
world = bpy.data.worlds.new("World")
world.use_nodes = True
scene.world = world
bpy.ops.object.camera_add(location=(0, 0, 3))
scene.camera = bpy.context.object
bpy.ops.render.render()
scene.render.engine = "VIBRT"
bpy.data.images["Render Result"].save_render(str(directory / "before.png"), scene=scene)
before = {
    "filepath": bpy.data.filepath,
    "dirty": bpy.data.is_dirty,
    "scene": scene.as_pointer(),
    "objects": len(bpy.data.objects),
    "materials": len(bpy.data.materials),
    "frame": scene.frame_current,
}

# Shader collection uses the real material exporter, without mesh or sky work.
depsgraph = bpy.context.evaluated_depsgraph_get()
with patch.object(
    exporter, "_export_mesh", side_effect=AssertionError("mesh export")
), patch.object(exporter, "_export_world", side_effect=AssertionError("sky export")):
    shaders = exporter.export_shader_scene(depsgraph)
assert shaders["aov_names"] == ["Noise", "Folded"]

captured = []
result = shader_precompile.CompileJob.result


def capture(job):
    stats = result(job)
    captured.append(stats)
    return stats


with patch.object(shader_precompile.CompileJob, "result", capture):
    assert bpy.ops.vibrt.precompile_shaders() == {"FINISHED"}
assert captured[0]["depsgraph_mode"] == "RENDER"
assert captured[0]["shader_ops"] & (1 << 3) == 0, "constant Math survived precompile"
assert shader_precompile.active_job is None
assert before == {
    "filepath": bpy.data.filepath,
    "dirty": bpy.data.is_dirty,
    "scene": bpy.context.scene.as_pointer(),
    "objects": len(bpy.data.objects),
    "materials": len(bpy.data.materials),
    "frame": scene.frame_current,
}
bpy.data.images["Render Result"].save_render(str(directory / "after.png"), scene=scene)
assert (directory / "before.png").read_bytes() == (directory / "after.png").read_bytes()

description, blobs, textures = exporter.export_scene_to_memory(
    depsgraph, serialize=False
)
assert shaders["materials"] == description["materials"]
warm = native.precompile_shaders(description)
for key in (
    "shader_ops",
    "max_closures",
    "shader_stack_size",
    "closure_types",
    "shader_output_count",
    "has_aovs",
    "has_extended_phases",
):
    assert warm[key] == captured[0][key], (key, warm, captured[0])
messages = []
aovs = {}
pixels, _ = native.render(
    description,
    [b.view(np.uint8) for b in blobs],
    {"spp": 1},
    messages.append,
    texture_arrays=textures,
    aov_outputs=aovs,
)
assert pixels.shape == (32, 32, 4) and np.isfinite(pixels).all()
assert pixels[:, :, :3].max() > 0
np.testing.assert_array_equal(aovs["Folded"][16, 16], [6.0, 6.0, 6.0, 1.0])
(directory / "results.json").write_text(
    json.dumps({"worker": captured[0], "warm": warm, "render_log": messages}, indent=2),
    encoding="utf-8",
)
print(
    f"[precompile] worker={captured[0]['seconds']:.2f}s, cached={warm['seconds']:.2f}s",
    flush=True,
)
print(
    "[precompile] operator, render depsgraph, cache reuse and Render Result isolation passed",
    flush=True,
)
# Precompilation is also useful before a camera has been placed in the scene.
scene.camera = None
objects = len(bpy.data.objects)
assert bpy.ops.vibrt.precompile_shaders() == {"FINISHED"}
assert scene.camera is None and len(bpy.data.objects) == objects
print("[precompile] camera-free scene passed", flush=True)
job = shader_precompile.start(scene, bpy.context.view_layer)
assert not bpy.ops.vibrt.precompile_shaders.poll()
assert bpy.ops.vibrt.cancel_precompile() == {"FINISHED"}
assert job.cancelled and job.process.poll() is not None
assert not job.directory.exists() and shader_precompile.active_job is None
print("[precompile] Cancel operator and duplicate-job guard passed", flush=True)
vibrt_blender.unregister()
