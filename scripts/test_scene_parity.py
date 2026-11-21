"""Small Cycles image regressions for the downloaded-scene failure cases.

blender -b --factory-startup --python-exit-code 1 --python scripts/test_scene_parity.py
Optional case names follow -- (coordinates, group_normal, lights, point_spot, volume, transparent, subsurface).
Artifacts and measured errors are written under target/scene-parity-tests/.
"""
from pathlib import Path
import json
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
import vibrt_blender
from vibrt_blender import exporter, vibrt_native

bpy.ops.wm.read_factory_settings(use_empty=True)
vibrt_blender.register()
OUT = ROOT / "target/scene-parity-tests"
OUT.mkdir(parents=True, exist_ok=True)
selected = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
rows = []


def setup():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    s = bpy.context.scene
    s.render.resolution_x, s.render.resolution_y = 144, 96
    s.render.resolution_percentage = 100
    s.render.pixel_aspect_x = s.render.pixel_aspect_y = 1
    s.cycles.samples = 2048
    s.cycles.use_denoising = s.cycles.use_adaptive_sampling = False
    s.cycles.max_bounces = 8
    s.cycles.diffuse_bounces = 6
    s.cycles.transmission_bounces = 12
    s.cycles.transparent_max_bounces = 16
    s.cycles.sample_clamp_direct = s.cycles.sample_clamp_indirect = 0
    s.cycles.blur_glossy = 0
    s.cycles.pixel_filter_type = "BOX"
    s.render.filter_size = 1
    s.render.use_compositing = False
    s.render.image_settings.file_format = "OPEN_EXR"
    s.render.image_settings.color_depth = "32"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
    s.cycles.device = "GPU"
    world = bpy.data.worlds.new("Parity")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0
    s.world = world
    bpy.ops.object.camera_add(location=(0, 0, 5))
    s.camera = bpy.context.object
    s.camera.data.type = "ORTHO"
    s.camera.data.ortho_scale = 3.6
    return s


def material(obj, shader="ShaderNodeEmission"):
    mat = bpy.data.materials.new("Parity")
    mat.use_nodes = True
    obj.data.materials.append(mat)
    tree = mat.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    node = tree.nodes.new(shader)
    tree.links.new(node.outputs[0], output.inputs["Surface"])
    return tree, node, output


def compare(name, s, tolerance, crops=None):
    s.render.engine = "CYCLES"
    s.render.filepath = str(OUT / f"{name}-cycles.exr")
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(s.render.filepath, check_existing=False)
    ref = np.empty(s.render.resolution_x * s.render.resolution_y * 4, np.float32)
    image.pixels.foreach_get(ref)
    ref = ref.reshape(s.render.resolution_y, s.render.resolution_x, 4)
    s.render.engine = "VIBRT"
    desc, blobs, textures = exporter.export_scene_to_memory(
        bpy.context.evaluated_depsgraph_get(), serialize=False)
    actual, _ = vibrt_native.render(desc, [b.view(np.uint8) for b in blobs],
                                    {"spp": s.cycles.samples}, texture_arrays=textures)
    assert np.isfinite(actual).all(), name
    np.save(OUT / f"{name}-cycles.npy", ref)
    np.save(OUT / f"{name}-vibrt.npy", actual)
    image = bpy.data.images.new(name, width=actual.shape[1], height=actual.shape[0], float_buffer=True)
    image.pixels.foreach_set(actual.ravel())
    image.save_render(str(OUT / f"{name}-vibrt.exr"), scene=s)
    if crops:
        errors = [float(abs(actual[crop].mean((0, 1))-ref[crop].mean((0, 1))).max())
                  for crop in crops]
        error = max(errors)
    else:
        error = float(abs(actual[..., :3]-ref[..., :3]).mean())
    row = dict(case=name, error=error, tolerance=tolerance,
               cycles_mean=ref[..., :3].mean((0, 1)).tolist(),
               vibrt_mean=actual[..., :3].mean((0, 1)).tolist())
    rows.append(row)
    (OUT / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("[scene-parity]", row, flush=True)


if not selected or "coordinates" in selected:
    s = setup()
    s.render.pixel_aspect_x, s.render.pixel_aspect_y = 1.3, 0.9
    s.camera.data.shift_x, s.camera.data.shift_y = 0.08, -0.04
    bpy.ops.object.empty_add(location=(-0.5, 0.2, 0.3))
    coordinate = bpy.context.object
    coordinate.rotation_euler = (0.2, 0.3, 0.5)
    coordinate.scale = (1.2, 0.7, 1.3)
    for x, z in [(-1, 0), (0, -1), (1, 0.5)]:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.45, location=(x, 0, z))
        tree, emission, _ = material(bpy.context.object)
        if x == -1:
            node = tree.nodes.new("ShaderNodeTexCoord")
            node.object = coordinate
            tree.links.new(node.outputs["Object"], emission.inputs["Color"])
            # A shader passed through a group input must stay the same closure.
            group_tree = bpy.data.node_groups.new("Shader passthrough", "ShaderNodeTree")
            group_tree.interface.new_socket(name="Shader", in_out="INPUT", socket_type="NodeSocketShader")
            group_tree.interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
            gi, go = group_tree.nodes.new("NodeGroupInput"), group_tree.nodes.new("NodeGroupOutput")
            group_tree.links.new(gi.outputs[0], go.inputs[0])
            group = tree.nodes.new("ShaderNodeGroup")
            group.node_tree = group_tree
            tree.links.new(emission.outputs[0], group.inputs[0])
            output = next(n for n in tree.nodes if n.type == "OUTPUT_MATERIAL")
            tree.links.new(group.outputs[0], output.inputs["Surface"])
        elif x == 0:
            node = tree.nodes.new("ShaderNodeNewGeometry")
            tree.links.new(node.outputs["Pointiness"], emission.inputs["Color"])
        else:
            node = tree.nodes.new("ShaderNodeTexImage")
            node.image = bpy.data.images.new("Missing parity image", 1, 1)
            node.image.source = "FILE"
            node.image.filepath = "//deliberately-missing-parity-image.png"
            tree.links.new(node.outputs["Color"], emission.inputs["Color"])
    compare("coordinates", s, 0.015)

if not selected or "group_normal" in selected:
    s = setup()
    s.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
    for poly in bpy.context.object.data.polygons:
        poly.use_smooth = True
    tree, _, output = material(bpy.context.object, "ShaderNodeBsdfDiffuse")
    group_tree = bpy.data.node_groups.new("Implicit normal", "ShaderNodeTree")
    socket = group_tree.interface.new_socket(name="Normal", in_out="INPUT", socket_type="NodeSocketVector")
    socket.hide_value = True
    group_tree.interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    gi, go = group_tree.nodes.new("NodeGroupInput"), group_tree.nodes.new("NodeGroupOutput")
    diffuse = group_tree.nodes.new("ShaderNodeBsdfDiffuse")
    reroute = group_tree.nodes.new("NodeReroute")
    group_tree.links.new(gi.outputs[0], reroute.inputs[0])
    group_tree.links.new(reroute.outputs[0], diffuse.inputs["Normal"])
    group_tree.links.new(diffuse.outputs[0], go.inputs[0])
    group = tree.nodes.new("ShaderNodeGroup")
    group.node_tree = group_tree
    tree.links.new(group.outputs[0], output.inputs["Surface"])
    compare("group_normal", s, 0.01)

if not selected or "lights" in selected:
    s = setup()
    bpy.ops.mesh.primitive_plane_add(size=100)
    tree, diffuse, _ = material(bpy.context.object, "ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.7, 0.7, 0.7, 1)
    bpy.ops.object.light_add(type="AREA", location=(0, 0.5, 2))
    light = bpy.context.object.data
    light.energy = 50
    light.shape, light.size, light.size_y = "ELLIPSE", 2, 1
    light.use_nodes = True
    tree = light.node_tree
    emission = next(n for n in tree.nodes if n.type == "EMISSION")
    emission.inputs["Color"].default_value = (0.7, 0.4, 0.2, 1)
    emission.inputs["Strength"].default_value = 3
    output = next(n for n in tree.nodes if n.type == "OUTPUT_LIGHT")
    path = tree.nodes.new("ShaderNodeLightPath")
    scale = tree.nodes.new("ShaderNodeMath")
    scale.operation = "DIVIDE"
    scale.inputs[1].default_value = 6
    tree.links.new(path.outputs["Ray Length"], scale.inputs[0])
    mix = tree.nodes.new("ShaderNodeMixShader")
    tree.links.new(scale.outputs[0], mix.inputs[0])
    tree.links.new(emission.outputs[0], mix.inputs[1])
    tree.links.new(mix.outputs[0], output.inputs["Surface"])
    compare("lights", s, 0.025)
    bpy.ops.mesh.primitive_cube_add(size=20)
    tree, absorption, output = material(bpy.context.object, "ShaderNodeVolumeAbsorption")
    tree.links.remove(output.inputs["Surface"].links[0])
    tree.links.new(absorption.outputs[0], output.inputs["Volume"])
    absorption.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1)
    absorption.inputs["Density"].default_value = 0.02
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0.5))
    material(bpy.context.object, "ShaderNodeBsdfTransparent")
    compare("lights_volume_transparent", s, 0.025)

if not selected or "point_spot" in selected:
    s = setup()
    bpy.ops.mesh.primitive_plane_add(size=100)
    tree, diffuse, _ = material(bpy.context.object, "ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.7, 0.7, 0.7, 1)
    for kind, x, falloff, color in [("POINT", -0.7, "Linear", (1, 0.2, 0.1, 1)),
                                    ("SPOT", 0.7, "Constant", (0.1, 0.5, 1, 1))]:
        bpy.ops.object.light_add(type=kind, location=(x, 0, 2))
        light = bpy.context.object.data
        light.energy = 20
        light.shadow_soft_size = 0
        light.use_nodes = True
        if kind == "SPOT":
            light.spot_size, light.spot_blend = 1.6, 1
        tree = light.node_tree
        emission = next(n for n in tree.nodes if n.type == "EMISSION")
        emission.inputs["Color"].default_value = color
        falloff_node = tree.nodes.new("ShaderNodeLightFalloff")
        falloff_node.inputs["Strength"].default_value = 4
        falloff_node.inputs["Smooth"].default_value = 0.5
        tree.links.new(falloff_node.outputs[falloff], emission.inputs["Strength"])
    compare("point_spot", s, 0.015)
    bpy.ops.mesh.primitive_cube_add(size=0.6, location=(0.1, 0, 0.7))
    material(bpy.context.object, "ShaderNodeBsdfDiffuse")
    for sphere in (True, False):
        for obj in s.objects:
            if obj.type == "LIGHT":
                obj.data.shadow_soft_size = 0.4
                obj.data.use_soft_falloff = not sphere
        compare("point_spot_sphere" if sphere else "point_spot_disk", s, 0.015)

if not selected or "volume" in selected:
    s = setup()
    bpy.ops.mesh.primitive_cube_add(size=1.5)
    tree, emission, output = material(bpy.context.object)
    tree.links.remove(output.inputs["Surface"].links[0])
    tree.links.new(emission.outputs[0], output.inputs["Volume"])
    emission.inputs["Color"].default_value = (0.2, 0.4, 0.8, 1)
    path = tree.nodes.new("ShaderNodeLightPath")
    tree.links.new(path.outputs["Is Camera Ray"], emission.inputs["Strength"])
    compare("volume", s, 0.01)
    s.camera.location.z = 0
    compare("volume_inside", s, 0.01)

if not selected or "transparent" in selected:
    s = setup()
    s.cycles.max_bounces = s.cycles.transmission_bounces = 0
    for z in (1, 2, 3, 4):
        bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, z))
        material(bpy.context.object, "ShaderNodeBsdfTransparent")
    bpy.ops.mesh.primitive_plane_add(size=10)
    tree, emission, _ = material(bpy.context.object)
    emission.inputs["Color"].default_value = (0.2, 0.4, 0.8, 1)
    path = tree.nodes.new("ShaderNodeLightPath")
    tree.links.new(path.outputs["Is Camera Ray"], emission.inputs["Strength"])
    compare("transparent", s, 0.001)

if not selected or "subsurface" in selected:
    s = setup()
    s.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1
    methods = ["BURLEY", "RANDOM_WALK", "RANDOM_WALK_SKIN"]
    for row in range(2):
        for col, method in enumerate(methods):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.4,
                                               location=(col-1, row-0.5, 0))
            for poly in bpy.context.object.data.polygons:
                poly.use_smooth = True
            tree, node, _ = material(bpy.context.object,
                "ShaderNodeBsdfPrincipled" if row else "ShaderNodeSubsurfaceScattering")
            if row:
                node.subsurface_method = method
                node.inputs["Subsurface Weight"].default_value = 1
                prefix, color = "Subsurface ", "Base Color"
            else:
                node.falloff = method
                prefix, color = "", "Color"
            node.inputs[color].default_value = (0.25, 0.5, 0.75, 1)
            node.inputs[prefix + "Radius"].default_value = (1, 0.5, 0.25)
            node.inputs[prefix + "Scale"].default_value = 0.8
    crops = [(slice(y-7, y+7), slice(x-7, x+7), slice(0, 3))
             for y in (28, 68) for x in (32, 72, 112)]
    compare("subsurface", s, 0.065, crops)

assert all(row["error"] < row["tolerance"] for row in rows), rows
