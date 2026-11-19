"""Compare shared C++ shader nodes and actual CUDA renders with Cycles 5.1.

blender -b --factory-startup --python-exit-code 1 --python scripts/test_shader_nodes.py
One atlas tests operations, dimensions, features and outputs in a single render.
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
from vibrt_blender import exporter, shader_graph, vibrt_native as native

bpy.ops.wm.read_factory_settings(use_empty=True)
vibrt_blender.register()
scene = bpy.context.scene
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.film_transparent = False
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.cycles.pixel_filter_type = "BOX"
scene.render.filter_size = 0.01
world = bpy.data.worlds.new("Black")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0,0,0,1)
scene.world = world
cases = []
errors = []
context = np.zeros((1, 128), np.float32)
context[0, :4] = (0.217, 0.371, 0.513, 1)
context[0, 4:8] = context[0, :4]
context[0, 16:20] = (0,0,1,1)
context[0, 24:28] = (0,0,1,1)
for base in (22,25,28):
    for axis in range(3):
        context[0, (base+axis)*4+axis] = 1


def case(kind, props=None, output=None, label=None, values=None):
    props = props or {}
    mat = bpy.data.materials.new(label or kind + str(props) + str(output))
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    node = tree.nodes.new(kind)
    for key, value in props.items():
        setattr(node, key, value)
    for index, sock in enumerate(node.inputs):
        if sock.type == "VECTOR":
            sock.default_value = (0.217, 0.371, 0.513)[:len(sock.default_value)]
        elif sock.type == "RGBA":
            sock.default_value = (0.17 + (index % 2)*0.4, 0.3, 0.71, 0.63)
    if values:
        for key, value in values.items():
            if isinstance(key, int) or node.inputs.get(key) is not None:
                node.inputs[key].default_value = value
    # Explicit coordinates make every tile uniform, allowing very tight comparisons.
    if node.inputs.get("Vector") is not None and kind.startswith("ShaderNodeTex"):
        coord = tree.nodes.new("ShaderNodeCombineXYZ")
        for i, v in enumerate((0.217, 0.371, 0.513)):
            coord.inputs[i].default_value = v
        tree.links.new(coord.outputs[0], node.inputs["Vector"])
    out = node.outputs[output] if output is not None else next(s for s in node.outputs if s.enabled)
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(out, emission.inputs["Color"])
    material_out = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs[0], material_out.inputs["Surface"])
    try:
        compiler = shader_graph.Compiler()
        index = compiler.input(emission.inputs["Color"])
        graph = compiler.graph({"base_color": index})
        expected = native.shader_pixels(graph, context)[0, 0, :3]
        cases.append((mat.name, mat, expected, graph))
    except Exception as exc:
        errors.append((mat.name, str(exc)))


for op in shader_graph.MATH:
    for values in ((0.37, 0.81, 0.29), (-0.4, 0.0, 0.5)):
        case("ShaderNodeMath", {"operation": op}, values=dict(enumerate(values)))
for op in shader_graph.VECTOR_MATH:
    case("ShaderNodeVectorMath", {"operation": op},
         values={0: (0.13, -0.37, 0.91), 1: (0.7, 0.2, -0.5), 2: (-0.2, 0.3, 0.5), 3: 0.71})
for dim in range(1,5):
    for kind in ("FBM", "MULTIFRACTAL", "HYBRID_MULTIFRACTAL", "RIDGED_MULTIFRACTAL", "HETERO_TERRAIN"):
        for output in ("Fac", "Color"):
            case("ShaderNodeTexNoise", {"noise_dimensions":f"{dim}D", "noise_type":kind}, output,
                 values={"W":0.19, "Detail":2.37, "Roughness":0.63, "Distortion":0.31})
    for feature in ("F1", "F2", "SMOOTH_F1", "DISTANCE_TO_EDGE", "N_SPHERE_RADIUS"):
        for metric in ("EUCLIDEAN", "MANHATTAN", "CHEBYCHEV", "MINKOWSKI"):
            case("ShaderNodeTexVoronoi", {"voronoi_dimensions":f"{dim}D", "feature":feature, "distance":metric, "normalize":True},
                 "Radius" if feature == "N_SPHERE_RADIUS" else "Distance", values={"W":0.19,"Detail":1.37,"Smoothness":0.6})
    for output in ("Value", "Color"):
        case("ShaderNodeTexWhiteNoise", {"noise_dimensions":f"{dim}D"}, output, values={"W":0.31})
for mode in shader_graph.BLENDS:
    case("ShaderNodeMixRGB", {"blend_type":mode}, values={0:0.37})
for mode in ("LINEAR", "QUADRATIC", "EASING", "DIAGONAL", "RADIAL", "QUADRATIC_SPHERE", "SPHERICAL"):
    case("ShaderNodeTexGradient", {"gradient_type":mode})
for mode in ("SIN", "SAW", "TRI"):
    for wave in ("BANDS", "RINGS"):
        case("ShaderNodeTexWave", {"wave_profile":mode,"wave_type":wave}, values={"Distortion":1.1,"Detail":2.3})
for kind in ("ShaderNodeTexBrick", "ShaderNodeTexChecker", "ShaderNodeTexMagic"):
    for output in ("Color", "Fac"):
        case(kind, output=output)
for dim in ("2D", "3D"):
    for output in ("Value", "Phase", "Intensity"):
        case("ShaderNodeTexGabor", {"gabor_type":dim}, output)
for kind in ("ShaderNodeGamma", "ShaderNodeInvert", "ShaderNodeHueSaturation", "ShaderNodeBrightContrast", "ShaderNodeRGBToBW", "ShaderNodeBlackbody", "ShaderNodeWavelength", "ShaderNodeValToRGB", "ShaderNodeRGBCurve", "ShaderNodeFloatCurve", "ShaderNodeVectorCurve"):
    case(kind)
for mode in ("POINT", "TEXTURE", "VECTOR", "NORMAL"):
    case("ShaderNodeMapping", {"vector_type":mode})
for mode in ("LINEAR", "STEPPED", "SMOOTHSTEP", "SMOOTHERSTEP"):
    case("ShaderNodeMapRange", {"interpolation_type":mode}, values={0:0.37, 5:4.0})
for mode in ("AXIS_ANGLE", "X_AXIS", "Y_AXIS", "Z_AXIS", "EULER_XYZ"):
    case("ShaderNodeVectorRotate", {"rotation_type":mode}, values={"Angle":0.73})
for output in ("Segment Coordinates", "Segment ID", "Segment Width", "Segment Rotation"):
    case("ShaderNodeRadialTiling", output=output, values={"Sides":5.5,"Roundness":0.37})
for mode in ("VECTOR", "POINT", "NORMAL"):
    case("ShaderNodeVectorTransform", {"vector_type":mode,"convert_from":"OBJECT","convert_to":"WORLD"})
assert not errors, json.dumps(errors, indent=2)

size = math.ceil(math.sqrt(len(cases)))
pixels_per_tile = 12
resolution = size*pixels_per_tile
scene.render.resolution_x = scene.render.resolution_y = resolution
cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.type = "PERSP"
cam.data.angle = math.pi/3
cam.location = (0,0,size/2/math.tan(cam.data.angle/2))
cam.rotation_euler = (0,0,0)
for i, (_, mat, _, _) in enumerate(cases):
    x, y = i%size-size/2+0.5, i//size-size/2+0.5
    mesh = bpy.data.meshes.new(f"tile_{i}")
    mesh.from_pydata([(x-0.5,y-0.5,0),(x+0.5,y-0.5,0),(x+0.5,y+0.5,0),(x-0.5,y+0.5,0)],[],[(0,1,2,3)])
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(mesh.name, mesh)
    scene.collection.objects.link(obj)

directory = ROOT / "target/shader-nodes"
directory.mkdir(parents=True, exist_ok=True)
scene.render.filepath = str(directory / "cycles.exr")
bpy.ops.render.render(write_still=True)
img = bpy.data.images.load(scene.render.filepath, check_existing=False)
pixels = np.empty(resolution*resolution*4, np.float32)
img.pixels.foreach_get(pixels)
reference = pixels.reshape(resolution,resolution,4)
failures = []
for i, (name, _, expected, _) in enumerate(cases):
    x,y = (i%size)*pixels_per_tile+pixels_per_tile//2, (i//size)*pixels_per_tile+pixels_per_tile//2
    actual = reference[y,x,:3]
    error = float(np.max(np.abs(expected-actual)/np.maximum(1,np.abs(actual))))
    if error > 2e-4:
        failures.append(dict(name=name, cpu=expected.tolist(), cycles=actual.tolist(), error=error))
(directory / "cpu-cycles.json").write_text(json.dumps({"cases":len(cases), "failures":failures}, indent=2))
(directory / "graphs.json").write_text(json.dumps([
    {"name":name,"graph":graph,"expected":expected.tolist()} for name,_,expected,graph in cases
]))
print(f"[shader-nodes] CPU/Cycles: {len(cases)} cases, {len(failures)} differences", flush=True)
for failure in failures[:12]:
    print(failure, flush=True)

if "--cpu-only" not in sys.argv:
    scene.render.engine = "VIBRT"
    exported, blobs, textures = exporter.export_scene_to_memory(bpy.context.evaluated_depsgraph_get(), serialize=False)
    missing = [m for m in exported["materials"] if "shader_graph" not in m]
    assert not missing, f"{len(missing)} atlas materials missing shader graph"
    pixels, _ = native.render(exported, [b.view(np.uint8) for b in blobs],
                              {"spp":1,"width":resolution,"height":resolution}, texture_arrays=textures)
    np.save(directory / "cuda.npy", pixels)
    # Native render buffers use the same bottom-up order as Blender images.
    gpu_failures = []
    for i, (name, _, expected, _) in enumerate(cases):
        x,y = (i%size)*pixels_per_tile+pixels_per_tile//2, (i//size)*pixels_per_tile+pixels_per_tile//2
        actual = pixels[y,x,:3]
        error = float(np.max(np.abs(expected-actual)/np.maximum(1,np.abs(expected))))
        if error > 5e-4:
            gpu_failures.append(dict(name=name, cpu=expected.tolist(), cuda=actual.tolist(), error=error))
    (directory / "cpu-cuda.json").write_text(json.dumps({"cases":len(cases),"failures":gpu_failures},indent=2))
    print(f"[shader-nodes] CPU/CUDA: {len(cases)} cases, {len(gpu_failures)} differences",flush=True)
    for failure in gpu_failures[:8]:
        print(failure,flush=True)
    assert not gpu_failures
assert not failures
print("[shader-nodes] PASS",flush=True)
