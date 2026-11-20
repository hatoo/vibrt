"""Compare spatial shader graphs, groups and closures against Cycles.

blender -b --factory-startup --python-exit-code 1 --python scripts/test_shader_graphs.py
Artifacts are written to target/shader-graphs; no user scene is modified.
"""
from pathlib import Path
import json
import math
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
import vibrt_blender
from vibrt_blender import exporter, vibrt_native as native

bpy.ops.wm.read_factory_settings(use_empty=True)
vibrt_blender.register()
scene = bpy.context.scene
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 16
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.cycles.pixel_filter_type = "BOX"
scene.render.filter_size = 0.01
world = bpy.data.worlds.new("Black")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
scene.world = world
cases = []


def material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    tree = mat.node_tree
    emit = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emit.outputs[0], output.inputs["Surface"])
    cases.append((name, mat))
    return tree, emit, output


image = bpy.data.images.new("RGBA fixture", width=2, height=2, alpha=True, float_buffer=True)
image.colorspace_settings.name = "Linear Rec.709"
image.alpha_mode = "STRAIGHT"
image.pixels.foreach_set([0.2, 0.4, 0.8, 0.25, 0.6, 0.3, 0.1, 0.75,
                         0.1, 0.7, 0.3, 0.5, 0.9, 0.2, 0.5, 1.0])
for interpolation in ("Closest", "Linear", "Cubic"):
    for output in ("Color", "Alpha"):
        tree, emit, _ = material(f"image-{interpolation}-{output}")
        tex = tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.interpolation = interpolation
        uv = tree.nodes.new("ShaderNodeUVMap")
        uv.uv_map = "Alternate"
        tree.links.new(uv.outputs[0], tex.inputs[0])
        tree.links.new(tex.outputs[output], emit.inputs["Color"])

for kind, output in (("ShaderNodeVertexColor", "Color"), ("ShaderNodeVertexColor", "Alpha"),
                     ("ShaderNodeAttribute", "Fac"), ("ShaderNodeObjectInfo", "Color"),
                     ("ShaderNodeObjectInfo", "Alpha"), ("ShaderNodeObjectInfo", "Object Index"),
                     ("ShaderNodeObjectInfo", "Material Index")):
    tree, emit, _ = material(f"{kind}-{output}")
    node = tree.nodes.new(kind)
    if kind == "ShaderNodeVertexColor":
        node.layer_name = "Tint"
    elif kind == "ShaderNodeAttribute":
        node.attribute_name = "Tint"
    tree.links.new(node.outputs[output], emit.inputs["Color"])

for kind, output in (("ShaderNodeTexCoord", "Generated"), ("ShaderNodeTexCoord", "UV"),
                     ("ShaderNodeTexCoord", "Object"), ("ShaderNodeTexCoord", "Normal"),
                     ("ShaderNodeTexCoord", "Camera"), ("ShaderNodeTexCoord", "Window"),
                     ("ShaderNodeNewGeometry", "Normal"), ("ShaderNodeNewGeometry", "Backfacing"),
                     ("ShaderNodeLightPath", "Is Camera Ray")):
    tree, emit, _ = material(f"{kind}-{output}")
    node = tree.nodes.new(kind)
    tree.links.new(node.outputs[output], emit.inputs["Color"])

for direction, axis in (("UV_MAP", "Z"), ("RADIAL", "X"), ("RADIAL", "Y"), ("RADIAL", "Z")):
    tree, emit, _ = material(f"tangent-{direction}-{axis}")
    node = tree.nodes.new("ShaderNodeTangent")
    node.direction_type, node.axis = direction, axis
    tree.links.new(node.outputs[0], emit.inputs["Color"])

for blend in (0.0, 0.5, 1.0):
    for output in ("Facing", "Fresnel"):
        tree, emit, _ = material(f"layer-weight-{blend}-{output}")
        node = tree.nodes.new("ShaderNodeLayerWeight")
        node.inputs["Blend"].default_value = blend
        tree.links.new(node.outputs[output], emit.inputs["Color"])

for alpha in (0.0, 0.5, 1.0):
    tree, emit, output = material(f"principled-alpha-{alpha}")
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0, 0, 0, 1)
    principled.inputs["Specular IOR Level"].default_value = 0
    principled.inputs["Emission Color"].default_value = (0.2, 0.4, 0.7, 1)
    value = tree.nodes.new("ShaderNodeValue")
    value.outputs[0].default_value = alpha
    tree.links.new(value.outputs[0], principled.inputs["Alpha"])
    tree.links.new(principled.outputs[0], output.inputs["Surface"])

for space in ("TANGENT", "OBJECT", "WORLD", "BLENDER_OBJECT", "BLENDER_WORLD"):
    for convention in ("OPENGL", "DIRECTX"):
        tree, emit, _ = material(f"normal-{space}-{convention}")
        node = tree.nodes.new("ShaderNodeNormalMap")
        node.space, node.convention = space, convention
        node.inputs["Color"].default_value = (0.3, 0.7, 0.9, 1)
        node.inputs["Strength"].default_value = 0.6
        tree.links.new(node.outputs[0], emit.inputs["Color"])

for kind in ("ShaderNodeMixShader", "ShaderNodeAddShader"):
    tree, emit, output = material(kind)
    shader = tree.nodes.new(kind)
    other = tree.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (0.2, 0.4, 0.7, 1)
    other.inputs["Color"].default_value = (0.8, 0.1, 0.3, 1)
    other.inputs["Strength"].default_value = 2
    tree.links.new(emit.outputs[0], shader.inputs[-2])
    tree.links.new(other.outputs[0], shader.inputs[-1])
    if kind == "ShaderNodeMixShader":
        coord = tree.nodes.new("ShaderNodeTexCoord")
        separate = tree.nodes.new("ShaderNodeSeparateXYZ")
        tree.links.new(coord.outputs["Generated"], separate.inputs[0])
        tree.links.new(separate.outputs["X"], shader.inputs[0])
    tree.links.new(shader.outputs[0], output.inputs["Surface"])

group = bpy.data.node_groups.new("Multiply with instance defaults", "ShaderNodeTree")
for name, direction in (("Input", "INPUT"), ("Output", "OUTPUT")):
    group.interface.new_socket(name=name, in_out=direction, socket_type="NodeSocketFloat")
gi = group.nodes.new("NodeGroupInput")
go = group.nodes.new("NodeGroupOutput")
math_node = group.nodes.new("ShaderNodeMath")
math_node.operation = "MULTIPLY"
math_node.inputs[1].default_value = 2
group.links.new(gi.outputs[0], math_node.inputs[0])
group.links.new(math_node.outputs[0], go.inputs[0])
tree, emit, _ = material("group-instance-defaults")
add = tree.nodes.new("ShaderNodeMath")
add.operation = "ADD"
for i, value in enumerate((0.17, 0.31)):
    instance = tree.nodes.new("ShaderNodeGroup")
    instance.node_tree = group
    instance.inputs[0].default_value = value
    tree.links.new(instance.outputs[0], add.inputs[i])
tree.links.new(add.outputs[0], emit.inputs["Color"])

for kind in ("ShaderNodeDisplacement", "ShaderNodeVectorDisplacement"):
    for space in (
        ("OBJECT", "WORLD")
        if kind == "ShaderNodeDisplacement"
        else ("TANGENT", "OBJECT", "WORLD")
    ):
        tree, emit, _ = material(f"{kind}-{space}")
        node = tree.nodes.new(kind)
        node.space = space
        node.inputs["Scale"].default_value = 0.4
        if kind == "ShaderNodeDisplacement":
            node.inputs["Height"].default_value = 0.9
        else:
            node.inputs["Vector"].default_value = (0.7, 0.2, 0.4, 1)
        tree.links.new(node.outputs[0], emit.inputs["Color"])

for map_name in ("UVMap", "Alternate"):
    for kind in ("ShaderNodeNormalMap", "ShaderNodeTangent"):
        tree, emit, _ = material(f"named-{kind}-{map_name}")
        node = tree.nodes.new(kind)
        node.uv_map = map_name
        if kind == "ShaderNodeNormalMap":
            node.inputs["Color"].default_value = (0.7, 0.7, 0.9, 1)
        tree.links.new(node.outputs[0], emit.inputs["Color"])

for attribute in ("Scalar", "Integer", "Boolean", "Vector", "Tint"):
    tree, emit, _ = material(f"attribute-{attribute}")
    node = tree.nodes.new("ShaderNodeAttribute")
    node.attribute_name = attribute
    tree.links.new(node.outputs["Color"], emit.inputs["Color"])

for kind in ("OBJECT", "VIEW_LAYER"):
    tree, emit, _ = material(f"attribute-{kind}")
    node = tree.nodes.new("ShaderNodeAttribute")
    node.attribute_type = kind
    node.attribute_name = "Fixture"
    tree.links.new(node.outputs["Color"], emit.inputs["Color"])
bpy.context.view_layer["Fixture"] = (0.2, 0.4, 0.7)

tree, emit, _ = material("nested-bump")
uv = tree.nodes.new("ShaderNodeTexCoord")
sep = tree.nodes.new("ShaderNodeSeparateXYZ")
tree.links.new(uv.outputs["UV"], sep.inputs[0])
first = tree.nodes.new("ShaderNodeBump")
second = tree.nodes.new("ShaderNodeBump")
first.inputs["Distance"].default_value = 0.1
second.inputs["Distance"].default_value = 0.15
tree.links.new(sep.outputs["X"], first.inputs["Height"])
tree.links.new(sep.outputs["Y"], second.inputs["Height"])
tree.links.new(first.outputs[0], second.inputs["Normal"])
tree.links.new(second.outputs[0], emit.inputs["Color"])

size = math.ceil(math.sqrt(len(cases)))
tile = 32
resolution = size * tile
scene.render.resolution_x = scene.render.resolution_y = resolution
camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
scene.collection.objects.link(camera)
scene.camera = camera
camera.data.angle = math.pi / 3
camera.location = (0, 0, size / 2 / math.tan(camera.data.angle / 2))
for i, (name, mat) in enumerate(cases):
    mesh = bpy.data.meshes.new(name)
    vertices = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)]
    if name.startswith("tangent-RADIAL"):
        # A flat Z bounding box makes radial X/Y tangents degenerate; avoid
        # comparing normalization of numerical noise in the reference renderer.
        vertices = [(x, y, 0.2*x + 0.1*y) for x, y, _ in vertices]
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
    for uv_name, values in (("UVMap", (0, 0, 1, 0, 1, 1, 0, 1)),
                            ("Alternate", (1, 0, 0, 0, 0, 1, 1, 1))):
        uv = mesh.uv_layers.new(name=uv_name)
        uv.data.foreach_set("uv", values)
    mesh.uv_layers.active_index = 0
    color = mesh.color_attributes.new(name="Tint", type="FLOAT_COLOR", domain="CORNER")
    color.data.foreach_set("color", (0.2, 0.4, 0.7, 0.35) * 4)
    for attr_name, kind, domain, value in (
        ("Scalar", "FLOAT", "POINT", 0.37),
        ("Integer", "INT", "FACE", 2),
        ("Boolean", "BOOLEAN", "POINT", True),
        ("Vector", "FLOAT_VECTOR", "CORNER", (0.2, 0.4, 0.7)),
    ):
        attr = mesh.attributes.new(attr_name, kind, domain)
        for element in attr.data:
            if kind == "FLOAT_VECTOR":
                element.vector = value
            else:
                element.value = value
    mesh.materials.append(mat)
    mat.pass_index = 3
    obj = bpy.data.objects.new(name, mesh)
    obj.location = (i % size - size / 2 + 0.5, i // size - size / 2 + 0.5, 0)
    obj.color = (0.13, 0.27, 0.53, 0.71)
    obj.pass_index = 2
    obj["Fixture"] = (0.2, 0.4, 0.7)
    scene.collection.objects.link(obj)

directory = ROOT / "target/shader-graphs"
directory.mkdir(parents=True, exist_ok=True)
scene.render.filepath = str(directory / "cycles.exr")
bpy.ops.render.render(write_still=True)
reference_image = bpy.data.images.load(scene.render.filepath, check_existing=False)
reference = np.empty(resolution * resolution * 4, np.float32)
reference_image.pixels.foreach_get(reference)
reference = reference.reshape(resolution, resolution, 4)
scene.render.engine = "VIBRT"
exported, blobs, textures = exporter.export_scene_to_memory(bpy.context.evaluated_depsgraph_get(), serialize=False)
assert len(exported["materials"]) == len(cases)
actual, _ = native.render(exported, [b.view(np.uint8) for b in blobs],
                          {"spp":16, "width":resolution, "height":resolution}, texture_arrays=textures)
np.save(directory / "cuda.npy", actual)
failures = []
for i, (name, _) in enumerate(cases):
    # Stay within one texel quadrant, away from triangle/texture discontinuities.
    x, y = i % size * tile + 8, i // size * tile + 8
    cpu, gpu = reference[y, x, :3], actual[y, x, :3]
    error = float(np.max(np.abs(cpu - gpu) / np.maximum(1, np.abs(cpu))))
    if error > 0.035:
        failures.append(dict(name=name, cycles=cpu.tolist(), cuda=gpu.tolist(), error=error))
(directory / "results.json").write_text(json.dumps({"cases":len(cases), "failures":failures}, indent=2))
print(f"[shader-graphs] {len(cases)} cases, {len(failures)} differences", flush=True)
for failure in failures:
    print(failure, flush=True)
assert not failures
