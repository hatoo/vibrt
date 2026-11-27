"""Cycles comparisons for light sampling, transparent shadows and adaptive AOVs."""
from pathlib import Path
import copy
import json
import math
import os
import re
import sys
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'blender'))
import vibrt_blender
from vibrt_blender import exporter, vibrt_native as native

bpy.ops.wm.read_factory_settings(use_empty=True)
vibrt_blender.register()
scene = bpy.context.scene
scene.render.resolution_x = scene.render.resolution_y = 16
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_depth = '32'
scene.cycles.device = 'CPU'
scene.cycles.samples = spp = int(os.environ.get('VIBRT_SAMPLING_TEST_SPP', '2048'))
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 2
scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0
scene.vibrt_clamp_indirect = 0
scene.cycles.pixel_filter_type = 'BOX'
world = bpy.data.worlds.new('Black')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
scene.world = world
bpy.ops.object.camera_add(location=(0, 0, 4))
scene.camera = bpy.context.object
scene.camera.data.angle = 0.01
bpy.ops.mesh.primitive_plane_add(size=100)
floor = bpy.context.object
material = bpy.data.materials.new('Diffuse')
material.use_nodes = True
material.node_tree.nodes.clear()
diffuse = material.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
diffuse.inputs['Color'].default_value = (0.7, 0.7, 0.7, 1)
out = material.node_tree.nodes.new('ShaderNodeOutputMaterial')
material.node_tree.links.new(diffuse.outputs[0], out.inputs['Surface'])
floor.data.materials.append(material)
directory = ROOT / 'target/reuse-sampling'
directory.mkdir(parents=True, exist_ok=True)
results = []


def compare(name, tolerance=0.025):
    if os.environ.get('VIBRT_TEST_FILTER', '') not in name:
        return
    scene.render.engine = 'CYCLES'
    scene.render.filepath = str(directory / f'{name}.exr')
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    pixels = np.empty(16 * 16 * 4, np.float32)
    image.pixels.foreach_get(pixels)
    reference = pixels.reshape(16, 16, 4)
    bpy.data.images.remove(image)
    scene.render.engine = 'VIBRT'
    desc, blobs, textures = exporter.export_scene_to_memory(bpy.context.evaluated_depsgraph_get(), serialize=False)
    actual, _ = native.render(desc, [a.view(np.uint8) for a in blobs],
                              {'spp': spp, 'denoise': False}, texture_arrays=textures)
    assert np.isfinite(actual).all(), name
    expected = reference[..., :3].mean((0, 1))
    got = actual[..., :3].mean((0, 1))
    error = float(np.max(abs(got-expected) / np.maximum(expected, 0.02)))
    results.append(dict(name=name, cycles=expected.tolist(), vibrt=got.tolist(), error=error))
    (directory / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    np.save(directory / f'{name}.npy', actual)
    print('[sampling]', results[-1], flush=True)
    assert error < tolerance, results[-1]
    return desc, blobs, textures


for kind, radius, shape, spread in [
    ('POINT', 0, None, None), ('POINT', 0.3, None, None),
    ('SPOT', 0, None, None), ('SPOT', 0.3, None, None),
    ('SUN', 0, None, None), ('SUN', 0.3, None, None),
    ('AREA', 0, 'RECTANGLE', math.pi), ('AREA', 0, 'RECTANGLE', math.pi/3),
    ('AREA', 0, 'DISK', math.pi/3), ('AREA', 0, 'ELLIPSE', 0.15),
    ('AREA', 0, 'RECTANGLE', 0),
]:
    data = bpy.data.lights.new('Test', kind)
    light = bpy.data.objects.new('Test', data)
    scene.collection.objects.link(light)
    light.location = (0.3, 0.2, 1)
    light.visible_camera = False
    data.energy = 20 if kind != 'SUN' else 2
    if kind in {'POINT', 'SPOT'}:
        data.shadow_soft_size = radius
    if kind == 'SPOT':
        data.spot_size, data.spot_blend = 1.2, 0.4
    if kind == 'SUN':
        data.angle = radius
        light.rotation_euler = (0.2, 0.4, 0)
    if kind == 'AREA':
        data.shape, data.size, data.size_y, data.spread = shape, 2, 1, spread
    compare(f'{kind}-{radius}-{shape}-{spread}')
    if kind == 'SPOT':
        light.scale = (0.6, 1.4, 0.7)
        compare(f'SPOT-{radius}-scaled')
    bpy.data.objects.remove(light, do_unlink=True)

# Direct colored transparency multiplies once per surface crossing.
data = bpy.data.lights.new('Shadow light', 'POINT')
data.energy, data.shadow_soft_size = 50, 0
light = bpy.data.objects.new('Shadow light', data)
scene.collection.objects.link(light)
light.location = (1, 0, 2)
for index in range(2):
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0.4 + index * 0.3))
    blocker = bpy.context.object
    blocker.visible_camera = False
    mat = bpy.data.materials.new('Colored transparent')
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    shader = mat.node_tree.nodes.new('ShaderNodeBsdfTransparent')
    shader.inputs['Color'].default_value = (0.8, 0.3, 0.1, 1)
    output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(shader.outputs[0], output.inputs['Surface'])
    blocker.data.materials.append(mat)
    compare(f'transparent-{index+1}')
    if index == 1:
        glass = mat.node_tree.nodes.new('ShaderNodeBsdfGlass')
        mat.node_tree.links.new(glass.outputs[0], output.inputs['Surface'])
        # Isolate direct shadow extinction; indirect glass caustics exercise a
        # different estimator and are covered by the surface closure suites.
        scene.cycles.max_bounces = 0
        compare('glass-shadow', 0.05)
        mix = mat.node_tree.nodes.new('ShaderNodeMixShader')
        opaque = mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
        mix.inputs[0].default_value = 0.25
        mat.node_tree.links.new(shader.outputs[0], mix.inputs[1])
        mat.node_tree.links.new(opaque.outputs[0], mix.inputs[2])
        mat.node_tree.links.new(mix.outputs[0], output.inputs['Surface'])
        compare('transparent-mix')
        scene.cycles.max_bounces = 2
for obj in list(scene.objects):
    if obj not in {scene.camera, floor, light}:
        bpy.data.objects.remove(obj, do_unlink=True)

# Convergence and AOV normalization must use actual per-pixel sample counts.
aov = bpy.context.view_layer.aovs.add()
aov.name, aov.type = 'AdaptiveValue', 'VALUE'
node = material.node_tree.nodes.new('ShaderNodeOutputAOV')
node.aov_name = aov.name
node.inputs['Value'].default_value = 0.37
desc, blobs, textures = exporter.export_scene_to_memory(bpy.context.evaluated_depsgraph_get(), serialize=False)
fixed_aovs, adaptive_aovs, logs = {}, {}, []
fixed, _ = native.render(desc, [a.view(np.uint8) for a in blobs], {'spp': 256},
                          texture_arrays=textures, aov_outputs=fixed_aovs)
desc['render'].update(use_adaptive_sampling=True, adaptive_threshold=0.01, adaptive_min_samples=16)
adaptive, _ = native.render(desc, [a.view(np.uint8) for a in blobs], {'spp': 256},
                             texture_arrays=textures, aov_outputs=adaptive_aovs, log_cb=logs.append)
line = next(s for s in logs if s.startswith('Adaptive sampling:'))
mean = float(re.search(r'mean ([\d.]+)', line)[1])
assert 16 <= mean < 256, line
np.testing.assert_allclose(adaptive, fixed, rtol=0.002, atol=0.001)
np.testing.assert_allclose(adaptive_aovs['AdaptiveValue'], fixed_aovs['AdaptiveValue'], atol=1e-5)
np.testing.assert_allclose(adaptive_aovs['AdaptiveValue'][..., :3], 0.37, atol=1e-5)
np.testing.assert_allclose(adaptive_aovs['AdaptiveValue'][..., 3], 1, atol=1e-5)
print('[sampling]', line, '; AOV normalization passed', flush=True)
