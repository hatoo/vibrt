"""Headless Blender checks for real displacement, image sampling and RNA isolation."""

from pathlib import Path
import sys
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
from vibrt_blender import exporter
from vibrt_blender.bin_writer import BinWriter
from vibrt_blender.shader_graph import UnsupportedNode

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_plane_add(size=2)
obj = bpy.context.object
obj.scale = (2, 3, 4)
obj.location = (1, 2, 3)
bpy.context.view_layer.update()
mat = bpy.data.materials.new("Displacement fixture")
mat.use_nodes = True
mat.displacement_method = "DISPLACEMENT"
obj.data.materials.append(mat)
tree = mat.node_tree
output = tree.nodes.get("Material Output")
original = np.array([v.co[:] for v in obj.data.vertices])
cases = 0
for space, expected in (
    ("OBJECT", (0.1, 0.2, 0.3)),
    ("WORLD", (0.05, 0.2 / 3, 0.075)),
    ("TANGENT", (0.1, 0.3, 0.2)),
):
    node = tree.nodes.new("ShaderNodeVectorDisplacement")
    node.space = space
    node.inputs["Scale"].default_value = 1
    node.inputs["Vector"].default_value = (0.1, 0.2, 0.3, 1)
    tree.links.new(node.outputs[0], output.inputs["Displacement"])
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    writer = BinWriter()
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    baseline = exporter._mesh_arrays_python(mesh, 1)["vertices"].reshape(-1, 3).copy()
    evaluated.to_mesh_clear()
    desc = exporter._export_mesh(evaluated, writer, [])
    positions = writer.mesh_blobs[desc["vertices"]].reshape(-1, 3)
    np.testing.assert_allclose(
        positions - baseline, np.tile(expected, (len(positions), 1)), atol=2e-6
    )
    np.testing.assert_array_equal(
        np.array([v.co[:] for v in obj.data.vertices]), original
    )
    tree.nodes.remove(node)
    cases += 1

# The CPU graph evaluates an image through the same sampler as CUDA.
node = tree.nodes.new("ShaderNodeDisplacement")
node.space = "OBJECT"
node.inputs["Scale"].default_value = 0.4
node.inputs["Midlevel"].default_value = 0.5
image = bpy.data.images.new("Constant height", width=2, height=2, float_buffer=True)
image.colorspace_settings.name = "Non-Color"
image.pixels.foreach_set([0.9, 0.9, 0.9, 1] * 4)
tex = tree.nodes.new("ShaderNodeTexImage")
tex.image = image
tree.links.new(tex.outputs["Color"], node.inputs["Height"])
tree.links.new(node.outputs[0], output.inputs["Displacement"])
bpy.context.view_layer.update()
writer = BinWriter()
desc = exporter._export_mesh(
    obj.evaluated_get(bpy.context.evaluated_depsgraph_get()), writer, []
)
positions = writer.mesh_blobs[desc["vertices"]].reshape(-1, 3)
np.testing.assert_allclose(positions[:, 2], 0.16, atol=2e-6)
np.testing.assert_array_equal(np.array([v.co[:] for v in obj.data.vertices]), original)
cases += 1

# BUMP must leave positions unchanged, including the legacy image-height path.
mat.displacement_method = "BUMP"
bpy.context.view_layer.update()
writer = BinWriter()
desc = exporter._export_mesh(
    obj.evaluated_get(bpy.context.evaluated_depsgraph_get()), writer, []
)
assert "displacement_tex" not in desc
np.testing.assert_array_equal(
    writer.mesh_blobs[desc["vertices"]].reshape(-1, 3)[:, 2], 0
)
cases += 1

# Inputs without pre-intersection meaning must fail before any mesh mutation.
mat.displacement_method = "DISPLACEMENT"
for kind, output_name in (
    ("ShaderNodeCameraData", "View Distance"),
    ("ShaderNodeAttribute", "Fac"),
):
    unsupported = tree.nodes.new(kind)
    if kind == "ShaderNodeAttribute":
        unsupported.attribute_type = "INSTANCER"
        unsupported.attribute_name = "height"
    tree.links.new(unsupported.outputs[output_name], node.inputs["Height"])
    bpy.context.view_layer.update()
    try:
        exporter._export_mesh(
            obj.evaluated_get(bpy.context.evaluated_depsgraph_get()), BinWriter(), []
        )
    except UnsupportedNode as exc:
        assert "Geometric displacement cannot evaluate" in str(exc)
    else:
        raise AssertionError(f"{kind} was silently evaluated with missing inputs")
    np.testing.assert_array_equal(
        np.array([v.co[:] for v in obj.data.vertices]), original
    )
    tree.nodes.remove(unsupported)
    cases += 1
print(f"[displacement] {cases} cases passed; original mesh preserved", flush=True)
