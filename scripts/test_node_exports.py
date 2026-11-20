"""Headless Blender checks for volume phases, hair/particle data and rejected modes."""

from pathlib import Path
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
from vibrt_blender import exporter, hair_export, material_export, shader_graph
from vibrt_blender.bin_writer import BinWriter

bpy.ops.wm.read_factory_settings(use_empty=True)
material = bpy.data.materials.new("Node exports")
material.use_nodes = True
tree = material.node_tree
tree.nodes.clear()
output = tree.nodes.new("ShaderNodeOutputMaterial")
volume = tree.nodes.new("ShaderNodeVolumeCoefficients")
volume.inputs["Scatter Coefficients"].default_value = (0.2, 0.3, 0.4)
tree.links.new(volume.outputs[0], output.inputs["Volume"])
for index, phase in enumerate(
    ("HENYEY_GREENSTEIN", "FOURNIER_FORAND", "DRAINE", "RAYLEIGH", "MIE")
):
    volume.phase = phase
    result = material_export._resolve_volume(output.inputs["Volume"])
    assert result["phase"] == index
    np.testing.assert_allclose(result["color"], (0.2, 0.3, 0.4))

other = tree.nodes.new("ShaderNodeVolumeCoefficients")
other.inputs["Scatter Coefficients"].default_value = (0.1, 0.1, 0.1)
add = tree.nodes.new("ShaderNodeAddShader")
tree.links.new(volume.outputs[0], add.inputs[0])
tree.links.new(other.outputs[0], add.inputs[1])
tree.links.new(add.outputs[0], output.inputs["Volume"])
volume.phase = other.phase = "RAYLEIGH"
assert material_export._resolve_volume(output.inputs["Volume"])["phase"] == 3
other.phase = "MIE"
try:
    material_export._resolve_volume(output.inputs["Volume"])
    raise AssertionError("Different volume phases must not silently become HG")
except ValueError as error:
    assert "different phase" in str(error)

hair = tree.nodes.new("ShaderNodeBsdfHairPrincipled")
hair.model = "HUANG"
try:
    shader_graph.compile_leaf(hair, BinWriter(), [])
    raise AssertionError("Unverified Huang mode was accepted")
except shader_graph.UnsupportedNode as error:
    assert "Huang" in str(error)
for kind in ("ShaderNodePointInfo", "ShaderNodeVolumeInfo"):
    node = tree.nodes.new(kind)
    try:
        shader_graph.Compiler().output(node.outputs[0])
        raise AssertionError(f"Missing geometry backend was accepted for {kind}")
    except shader_graph.UnsupportedNode as error:
        assert "not available" in str(error)

# Legacy particle object instances expose particle data through the depsgraph.
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.05, location=(10, 0, 0))
prototype = bpy.context.object
bpy.ops.mesh.primitive_plane_add(size=2)
emitter = bpy.context.object
bpy.ops.object.particle_system_add()
settings = emitter.particle_systems[0].settings
settings.count = 4
settings.frame_start = settings.frame_end = 1
settings.lifetime = 100
settings.render_type = "OBJECT"
settings.instance_object = prototype
bpy.context.scene.frame_set(10)
depsgraph = bpy.context.evaluated_depsgraph_get()
keys = {
    "particle:" + name
    for name in (
        "Index",
        "Age",
        "Lifetime",
        "Location",
        "Size",
        "Velocity",
        "Angular Velocity",
        "Random",
    )
}
instances = []
for instance in depsgraph.object_instances:
    if instance.is_instance and instance.particle_system is not None:
        attributes = exporter._export_object_attributes(
            instance.object, instance, keys, bpy.context.scene
        )
        assert keys <= attributes.keys(), attributes
        assert attributes["particle:Lifetime"][0] == 100
        assert attributes["particle:Age"][0] == 9
        assert np.isfinite(list(attributes.values())).all()
        instances.append(attributes["particle:Index"][0])
assert sorted(instances) == [0, 1, 2, 3], instances

# Hair attributes follow the emitted ribbon's corners, including its full width.
bpy.ops.mesh.primitive_plane_add(size=2)
emitter = bpy.context.object
emitter.data.materials.append(material)
bpy.ops.object.particle_system_add()
settings = emitter.particle_systems[0].settings
settings.type = "HAIR"
settings.count = 2
settings.hair_length = 1
settings.hair_step = 2
settings.radius_scale = 0.01
settings.root_radius = 1
settings.tip_radius = 0.5
bpy.context.scene.frame_set(1)
depsgraph = bpy.context.evaluated_depsgraph_get()
keys = {
    "hair:" + name
    for name in ("Is Strand", "Intercept", "Length", "Thickness", "Random")
}
writer = BinWriter()
result = hair_export.export_hair(
    emitter,
    emitter.evaluated_get(depsgraph),
    bpy.context.scene.evaluated_get(depsgraph),
    writer,
    lambda _: 0,
    emitter.name,
    {material.name: keys},
)
assert result is not None
mesh, _ = result
attributes = {
    name: writer.mesh_blobs[index].reshape(-1, 4)
    for name, index in mesh["attributes"].items()
}
assert keys == attributes.keys()
assert np.all(attributes["hair:Is Strand"][:, 0] == 1)
np.testing.assert_allclose(
    [
        attributes["hair:Intercept"][:, 0].min(),
        attributes["hair:Intercept"][:, 0].max(),
    ],
    [0, 1],
)
np.testing.assert_allclose(attributes["hair:Length"][:, 0], 1, atol=1e-5)
np.testing.assert_allclose(
    [
        attributes["hair:Thickness"][:, 0].min(),
        attributes["hair:Thickness"][:, 0].max(),
    ],
    [0.005, 0.01],
    atol=1e-7,
)
assert np.all(
    (attributes["hair:Random"][:, 0] >= 0) & (attributes["hair:Random"][:, 0] < 1)
)
print(
    "[node-exports] 5 volume phases, phase mixing, unsupported modes, particles and hair passed",
    flush=True,
)
