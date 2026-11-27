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
result = shader_graph.compile_leaf(hair, BinWriter(), [])
assert result['closure_type'] == 6
assert {'hair_roughness_v', 'hair_coat', 'hair_transmission', 'hair_secondary'} <= set(result['shader_graph']['outputs'])
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

# Hair attributes follow native curve keys, including the authored full width.
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
settings.use_close_tip = False
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
assert len(result) == 1
mesh, _ = result[0]
assert 'curve_radii' in mesh
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
# Child shape is Blender's evaluated cache, including curl, rather than an
# approximation reconstructed from parent paths.
settings.child_type = "SIMPLE"
settings.child_percent = 3
settings.kink = "CURL"
settings.kink_amplitude = 0.05
settings.kink_frequency = 2
bpy.context.scene.frame_set(2)
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = emitter.evaluated_get(depsgraph)
system = evaluated.particle_systems[0]
children = len(system.child_particles)
assert children > 0
writer = BinWriter()
pairs = hair_export.export_hair(emitter, evaluated, bpy.context.scene, writer,
                                lambda _: 0, emitter.name, depsgraph=depsgraph)
mesh, _ = pairs[0]
points = writer.mesh_blobs[mesh['vertices']].reshape(-1, 3)
steps = (1 << settings.display_step) + 1
assert len(points) == children * steps
expected = np.asarray([system.co_hair(evaluated, particle_no=len(system.particles), step=i)
                       for i in range(steps)], np.float32)
inverse = np.asarray(evaluated.matrix_world.inverted(), np.float32)
np.testing.assert_allclose(points[:steps], expected @ inverse[:3, :3].T + inverse[:3, 3], atol=1e-6)
# Hiding the emitter surface must retain its evaluated hair and visibility.
import vibrt_blender
vibrt_blender.register()
hair_scene = bpy.data.scenes.new("Hair without emitter")
bpy.context.window.scene = hair_scene
bpy.ops.mesh.primitive_plane_add(size=2)
emitter = bpy.context.object
emitter.show_instancer_for_render = False
emitter.visible_camera = False
bpy.ops.object.particle_system_add()
settings = emitter.particle_systems[0].settings
settings.type, settings.count, settings.hair_length = "HAIR", 2, 1
settings.radius_scale = 0.01
bpy.ops.object.camera_add(location=(0, 0, 4))
hair_scene.camera = bpy.context.object
hair_scene.frame_set(1)
desc, _, _ = exporter.export_scene_to_memory(bpy.context.evaluated_depsgraph_get(), serialize=False)
assert len(desc['meshes']) == 1 and 'curve_radii' in desc['meshes'][0]
assert len(desc['objects']) == 1 and not desc['objects'][0]['camera_visible']
print(
    "[node-exports] 5 volume phases, phase mixing, unsupported modes, particles and hair passed",
    flush=True,
)
