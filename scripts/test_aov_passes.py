"""Exercise real Blender RenderResult pass registration and final AOV writes."""

from pathlib import Path
import sys
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
import vibrt_blender
from vibrt_blender.progressive import RenderResultDisplay

bpy.ops.wm.read_factory_settings(use_empty=True)
vibrt_blender.register()
scene = bpy.context.scene
scene.render.engine = "VIBRT"
scene.vibrt_spp = 4
scene.vibrt_denoise = False
scene.render.resolution_x = scene.render.resolution_y = 32
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
bpy.ops.mesh.primitive_plane_add(size=2)
mat = bpy.data.materials.new("AOV fixture")
mat.use_nodes = True
bpy.context.object.data.materials.append(mat)
tree = mat.node_tree
tree.nodes.clear()
emit = tree.nodes.new("ShaderNodeEmission")
emit.inputs["Color"].default_value = (0.2, 0.4, 0.7, 1)
out = tree.nodes.new("ShaderNodeOutputMaterial")
tree.links.new(emit.outputs[0], out.inputs["Surface"])
for name, kind, value in (
    ("CustomColor", "COLOR", (0.13, 0.27, 0.53, 1)),
    ("CustomValue", "VALUE", 0.37),
):
    aov = bpy.context.view_layer.aovs.add()
    aov.name = name
    aov.type = kind
    node = tree.nodes.new("ShaderNodeOutputAOV")
    node.aov_name = name
    node.inputs["Color" if kind == "COLOR" else "Value"].default_value = value
aov = bpy.context.view_layer.aovs.add()
aov.name = "BumpedNormal"
aov.type = "COLOR"
node = tree.nodes.new("ShaderNodeOutputAOV")
node.aov_name = aov.name
geometry = tree.nodes.new("ShaderNodeNewGeometry")
separate = tree.nodes.new("ShaderNodeSeparateXYZ")
bump = tree.nodes.new("ShaderNodeBump")
bump.inputs["Distance"].default_value = 1
tree.links.new(geometry.outputs["Position"], separate.inputs[0])
tree.links.new(separate.outputs["X"], bump.inputs["Height"])
tree.links.new(bump.outputs["Normal"], node.inputs["Color"])
# A reserved pass name is invalid and must never overwrite the beauty pass.
invalid = bpy.context.view_layer.aovs.add()
invalid.name = "Combined"
invalid_node = tree.nodes.new("ShaderNodeOutputAOV")
invalid_node.aov_name = "Combined"
invalid_node.inputs["Color"].default_value = (1, 0, 0, 1)
bpy.ops.object.camera_add(location=(0, 0, 3))
scene.camera = bpy.context.object
scene.camera.data.angle = 1.2
checked = []
original_finish = RenderResultDisplay.finish


def finish(self, pixels, depth, mist, aovs=None):
    original_finish(self, pixels, depth, mist, aovs)
    assert "Combined" not in aovs
    combined = np.asarray(self.passes["Combined"].rect).reshape(
        self.height, self.width, 4
    )
    np.testing.assert_allclose(combined[16, 16], pixels[16, 16], atol=1e-6)
    for name, channels, expected in (
        ("CustomColor", 4, (0.13, 0.27, 0.53, 1)),
        ("CustomValue", 1, (0.37,)),
        ("BumpedNormal", 4, (-(2**-0.5), 0, 2**-0.5, 1)),
    ):
        rpass = self.passes[name]
        assert rpass.channels == channels
        data = np.asarray(rpass.rect).reshape(self.height, self.width, channels)
        np.testing.assert_allclose(
            data[14:18, 14:18].mean(axis=(0, 1)), expected, atol=5e-5
        )
    assert pixels[16, 16, 3] > 0.99 and pixels[0, 0, 3] < 0.01
    checked.append(True)


RenderResultDisplay.finish = finish
bpy.ops.render.render()
assert checked, "Render engine did not finish its AOV passes"
print(
    "[aov-passes] Color, Value, bump dependencies and transparent film passed",
    flush=True,
)
