"""GPU regression for opaque instances sharing geometry with transparent ones.

Run with Blender --background --factory-startup --python-exit-code 1 --python
scripts/test_anyhit_render.py. Before rebuilding, add -- --save-baseline;
--native PATH can load a saved extension in a separate Blender process.
"""

import copy
import importlib.util
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
import vibrt_blender
from vibrt_blender import exporter

if "--native" in sys.argv:
    path = Path(sys.argv[sys.argv.index("--native") + 1]).resolve()
    spec = importlib.util.spec_from_file_location("vibrt_blender.vibrt_native", path)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    vibrt_blender.vibrt_native = native
else:
    from vibrt_blender import vibrt_native as native

vibrt_blender.register()
scene = bpy.context.scene
scene.render.engine = "VIBRT"
cube = bpy.data.objects["Cube"]
cube.data.materials.clear()
cube.data.materials.append(bpy.data.materials.new("Blocker"))
bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -1))
bpy.context.object.data.materials.append(bpy.data.materials.new("Receiver"))
description, arrays, textures = exporter.export_scene_to_memory(
    bpy.context.evaluated_depsgraph_get(), serialize=False
)
# This fixture always uses linear pixels; the alpha mask has both holes and solids.
textures.append(np.array([1, 1, 1, 0, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 0], np.float32))
description["textures"].append(dict(width=2, height=2, channels=4,
                                    colorspace="linear", array_index=len(textures)-1))
cube_id = next(i for i, obj in enumerate(description["objects"])
               if len(arrays[description["meshes"][obj["mesh"]]["indices"]]) == 36)
material_id = description["objects"][cube_id]["material"]
opaque = dict(base_color=[0.6, 0.2, 0.1], roughness=0.5)
transparent = dict(opaque, alpha_blend=0.3)
cases = {
    "opaque": opaque,
    "alpha": transparent,
    "cutout": dict(opaque, alpha_threshold=0.5,
                   base_color_tex=len(description["textures"])-1),
    "glass": dict(opaque, transmission=1, roughness=0.1),
    "volume": dict(opaque, volume_only=True,
                   volume=dict(density=0.1, color=[0.5, 0.5, 0.5])),
    "mixed": dict(opaque, mix_fac=0.5, secondary=transparent),
}
outputs = {}
for name, material in cases.items():
    data = copy.deepcopy(description)
    data["materials"][material_id] = material
    pixels, depth = native.render(
        data, [a.view(np.uint8) for a in arrays],
        dict(spp=16, width=96, height=54), texture_arrays=textures,
    )
    assert np.isfinite(pixels).all(), name
    outputs[name] = pixels
    outputs[name + "-depth"] = depth

data = copy.deepcopy(description)
data["materials"][material_id] = opaque
data["materials"].append(transparent)
other = copy.deepcopy(data["objects"][cube_id])
other["transform"][3] -= 2.5
other["material"] = len(data["materials"]) - 1
data["objects"].append(other)
pixels, depth = native.render(data, [a.view(np.uint8) for a in arrays],
                              dict(spp=16, width=96, height=54), texture_arrays=textures)
outputs["shared-mesh"] = pixels
outputs["shared-mesh-depth"] = depth

mesh = data["meshes"][data["objects"][cube_id]["mesh"]]
indices = np.arange(len(arrays[mesh["indices"]]) // 3, dtype=np.uint32) % 2
mesh["material_indices"] = len(arrays)
arrays.append(indices)
for obj in [data["objects"][cube_id], other]:
    obj["materials"] = [material_id, len(data["materials"]) - 1]
pixels, depth = native.render(data, [a.view(np.uint8) for a in arrays],
                              dict(spp=16, width=96, height=54), texture_arrays=textures)
outputs["material-slots"] = pixels
outputs["material-slots-depth"] = depth

for name in (*cases, "shared-mesh", "material-slots"):
    assert np.isfinite(outputs[name]).all(), name
    if name != "opaque":
        assert np.any(outputs[name] != outputs["opaque"]), f"{name} fixture has no effect"

directory = ROOT / "target/anyhit-test"
directory.mkdir(parents=True, exist_ok=True)
if "--save-baseline" in sys.argv:
    np.savez(directory / "baseline.npz", **outputs)
    print("[anyhit-test] saved baseline for 8 scenes", flush=True)
else:
    baseline = directory / "baseline.npz"
    if not baseline.exists():
        raise RuntimeError("Create a baseline with the original extension first")
    with np.load(baseline) as reference:
        assert set(reference.files) == set(outputs)
        for name, value in outputs.items():
            np.testing.assert_allclose(value, reference[name], rtol=1e-6, atol=1e-6,
                                       err_msg=name)
    np.savez(directory / "after.npz", **outputs)
    print("[anyhit-test] 8 scenes match original pixels and depth", flush=True)
