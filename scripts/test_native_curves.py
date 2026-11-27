"""Compare Chiang/Huang on native curves with Cycles; exercise particle caches.

blender -b --factory-startup --python-exit-code 1 --python scripts/test_native_curves.py
"""
from pathlib import Path
import json
import os
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
scene.render.resolution_x, scene.render.resolution_y = 96, 64
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_depth = '32'
scene.cycles.device = 'CPU'
scene.cycles.samples = spp = int(os.environ.get('VIBRT_CURVE_TEST_SPP', '2048'))
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.cycles.pixel_filter_type = 'BOX'
scene.cycles.filter_width = 1
scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0
scene.vibrt_clamp_indirect = 0
world = bpy.data.worlds.new('White')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
scene.world = world
bpy.ops.object.camera_add(location=(0, 0, 4))
scene.camera = bpy.context.object
scene.camera.data.type = 'ORTHO'
scene.camera.data.ortho_scale = 3
for index, (model, aspect) in enumerate([('CHIANG', 1), ('HUANG', 1), ('HUANG', 0.6), ('HUANG', 1.4)]):
    mat = bpy.data.materials.new(f'{model}-{aspect}')
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    node = mat.node_tree.nodes.new('ShaderNodeBsdfHairPrincipled')
    node.model = model
    node.parametrization = 'COLOR'
    node.inputs['Color'].default_value = (0.5, 0.25, 0.1, 1)
    node.inputs['Roughness'].default_value = 0.3
    if model == 'CHIANG':
        node.inputs['Radial Roughness'].default_value = 0.4
    else:
        node.inputs['Aspect Ratio'].default_value = aspect
    output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(node.outputs[0], output.inputs['Surface'])
    data = bpy.data.hair_curves.new(mat.name)
    data.add_curves([2])
    x = (index - 1.5) * 0.65
    data.position_data.foreach_set('vector', [x, -0.8, 0, x, 0.8, 0])
    radius = data.attributes.new('radius', 'FLOAT', 'POINT')
    radius.data.foreach_set('value', [0.11, 0.11])
    data.materials.append(mat)
    obj = bpy.data.objects.new(mat.name, data)
    scene.collection.objects.link(obj)

directory = ROOT / 'target/native-curves'
directory.mkdir(parents=True, exist_ok=True)
scene.render.engine = 'CYCLES'
scene.render.filepath = str(directory / 'cycles.exr')
bpy.ops.render.render(write_still=True)
image = bpy.data.images.load(scene.render.filepath, check_existing=False)
reference = np.empty(96 * 64 * 4, np.float32)
image.pixels.foreach_get(reference)
reference = reference.reshape(64, 96, 4)
scene.render.engine = 'VIBRT'
desc, blobs, textures = exporter.export_scene_to_memory(bpy.context.evaluated_depsgraph_get(), serialize=False)
assert len(desc['meshes']) == 4 and all('curve_radii' in m for m in desc['meshes'])
assert {m['closure_type'] for m in desc['materials']} >= {5, 6}
(directory / 'scene.json').write_text(json.dumps(desc, indent=2), encoding='utf-8')
actual, _ = native.render(desc, [a.view(np.uint8) for a in blobs], {'spp': spp, 'denoise': False}, texture_arrays=textures)
assert np.isfinite(actual).all()
np.save(directory / 'vibrt.npy', actual)
np.save(directory / 'cycles.npy', reference)
results = []
for i in range(4):
    x = round(48 + (i - 1.5) * 0.65 * 32)
    region = (slice(12, 52), slice(x-2, x+2), slice(0, 3))
    expected = reference[region].mean(axis=(0, 1))
    got = actual[region].mean(axis=(0, 1))
    error = float(np.max(np.abs(got-expected) / np.maximum(expected, 0.05)))
    tile = (slice(8, 56), slice(x-9, x+9))
    coverage_error = float(abs(actual[tile][..., 3].sum() - reference[tile][..., 3].sum())
                           / reference[tile][..., 3].sum())
    energy_error = float(abs(actual[tile][..., :3].sum() - reference[tile][..., :3].sum())
                         / reference[tile][..., :3].sum())
    results.append(dict(case=i, cycles=expected.tolist(), vibrt=got.tolist(), relative_error=error,
                        coverage_error=coverage_error, energy_error=energy_error))
    print('[curves]', results[-1], flush=True)
(directory / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
for name, pixels in [('cycles', reference), ('vibrt', actual)]:
    image = bpy.data.images.new(name, width=96, height=64, alpha=True, float_buffer=True)
    image.pixels.foreach_set(pixels.ravel())
    scene.render.image_settings.file_format = 'PNG'
    image.save_render(str(directory / f'{name}.png'), scene=scene)
    bpy.data.images.remove(image)
assert max(r['relative_error'] for r in results) < 0.08, results
assert max(r['coverage_error'] for r in results) < 0.03, results
assert max(r['energy_error'] for r in results) < 0.08, results
