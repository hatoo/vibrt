"""Compare native Blender memory reads with the public RNA reference path.

Run in a separate process:
  blender -b --factory-startup --python-exit-code 1 --python scripts/test_native_scene.py
Pass -- --scene to check every mesh in the loaded .blend instead of fixtures.
No scene, preferences, or recovery files are saved.
"""

import json
import os
from pathlib import Path
import sys
import time

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
from vibrt_blender import exporter, material_export, vibrt_native as native


def compare_mesh(mesh, slots=2, color=None):
    mesh.calc_loop_triangles()
    expected = exporter._mesh_arrays_python(mesh, slots, color)
    actual = native.read_mesh(mesh, slots, color)
    assert actual is not None, (mesh.name, color, "native path was skipped")
    assert expected.keys() == actual.keys(), (expected.keys(), actual.keys())
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"{mesh.name}: {key}")
        assert actual[key].flags.c_contiguous
        assert actual[key].dtype == expected[key].dtype
    return actual


def mesh_fixtures():
    mesh = bpy.data.meshes.new("native_sync_fixture")
    # A non-planar ngon plus a triangle, shared points, and a loose point/edge.
    mesh.from_pydata(
        [(0, 0, 0), (2, 0, 0), (3, 1, 0.2), (1, 3, 0), (-1, 1, 0), (0, 0, 2)],
        [(0, 5)], [(0, 1, 2, 3, 4), (1, 0, 5)],
    )
    compare_mesh(mesh, slots=0)
    mesh.polygons[0].material_index = 1
    mesh.polygons[1].use_smooth = True
    uv = mesh.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", np.linspace(-2, 3, len(mesh.loops) * 2, dtype=np.float32))
    mesh.normals_split_custom_set([(0.1, 0.2, 1.0)] * len(mesh.loops))
    compare_mesh(mesh)
    for domain in ("POINT", "CORNER"):
        attr = mesh.color_attributes.new(name=f"color_{domain}", type="FLOAT_COLOR", domain=domain)
        attr.data.foreach_set("color", np.linspace(0, 2, len(attr.data) * 4, dtype=np.float32))
        compare_mesh(mesh, color=attr.name)
    compare_mesh(mesh, color="missing")
    byte = mesh.color_attributes.new(name="byte_color", type="BYTE_COLOR", domain="CORNER")
    byte.data.foreach_set("color", np.linspace(0, 1, len(byte.data) * 4, dtype=np.float32))
    compare_mesh(mesh, color=byte.name)

    obj = bpy.data.objects.new("native_sync_fixture", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new("triangulate", "TRIANGULATE")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for iteration in range(3):
        obj_eval = obj.evaluated_get(depsgraph)
        evaluated = obj_eval.to_mesh()
        try:
            arrays = compare_mesh(evaluated, color="color_CORNER")
            saved = {key: value.copy() for key, value in arrays.items()}
        finally:
            obj_eval.to_mesh_clear()
        # Returned arrays own their data after the evaluated mesh is released.
        for key in saved:
            np.testing.assert_array_equal(arrays[key], saved[key])
        mesh.vertices[0].co.x += 0.125
        mesh.update()
        bpy.context.view_layer.update()
    empty = bpy.data.meshes.new("empty")
    compare_mesh(empty, slots=0)
    try:
        native.read_mesh(object(), 0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-Blender objects must be rejected")


def scene_meshes():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    count = 0
    native_time = reference_time = 0.0
    for obj in depsgraph.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.to_mesh()
        try:
            mesh.calc_loop_triangles()
            start = time.perf_counter()
            expected = exporter._mesh_arrays_python(mesh, len(obj.material_slots))
            reference_time += time.perf_counter() - start
            start = time.perf_counter()
            actual = native.read_mesh(mesh, len(obj.material_slots))
            native_time += time.perf_counter() - start
            assert actual is not None, mesh.name
            assert actual.keys() == expected.keys()
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"{obj.name}: {key}")
            count += 1
        finally:
            obj.to_mesh_clear()
    assert count > 0
    print(f"[native-scene-test] {count} meshes exact; reference={reference_time:.4f}s native={native_time:.4f}s")


def pixels():
    rng = np.random.default_rng(20261119)
    original = os.environ.get("VIBRT_PYTHON_EXPORT")
    os.environ["VIBRT_PYTHON_EXPORT"] = "1"
    try:
        values = np.concatenate((np.array([-1, 0, 0.04045, 1, 8], dtype=np.float32),
                                 rng.uniform(-0.01, 4, 100000).astype(np.float32)))
        with np.errstate(invalid="ignore"):
            reference = material_export._srgb_to_linear_np(values)
        np.testing.assert_allclose(native.srgb_to_linear(values), reference, rtol=4e-7, atol=1e-8)
        for shape in ((1, 1, 3), (1, 7, 4), (8, 1, 3), (5, 7, 3), (63, 81, 4)):
            array = rng.uniform(-2, 4, shape).astype(np.float32)
            for height, width in ((1, 1), (3, 2), (13, 17), (100, 128)):
                reference = material_export._resample_bilinear(array, height, width)
                actual = native.resample_bilinear(array, height, width)
                np.testing.assert_array_equal(actual, reference)
        for height, width in ((0, 1), (1, 0), (-1, 3)):
            try:
                native.resample_bilinear(np.ones((2, 2, 3), dtype=np.float32), height, width)
            except (ValueError, OverflowError):
                pass
            else:
                raise AssertionError("invalid dimensions accepted")
    finally:
        if original is None:
            os.environ.pop("VIBRT_PYTHON_EXPORT", None)
        else:
            os.environ["VIBRT_PYTHON_EXPORT"] = original


def scene_description():
    # These fail before GPU initialization; exercise the same schema via both
    # dict and JSON, including nested options, tagged enums and missing fields.
    scene = {
        "version": 999,
        "render": {"width": 1, "height": 1, "spp": 1, "max_depth": 1},
        "camera": {"transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], "fov_y_rad": 1.0},
        "materials": [{"base_color": [0.2, 0.3, 0.4], "color_graph": {"nodes": [
            {"type": "const", "rgb": [1, 0, 0]},
            {"type": "mix", "a": 0, "b": 0, "fac": {"node": 0}},
        ]}}],
        "world": {"type": "constant", "color": [0, 0, 0]},
        "world_volume": None,
        "ignored": "field",
    }
    for value in (scene, json.dumps(scene)):
        try:
            native.render(value, [], {})
        except RuntimeError as error:
            assert "unsupported scene.json version: 999" in str(error), str(error)
        else:
            raise AssertionError("invalid scene version accepted")
    scene["camera"]["transform"] = [1, 2]
    try:
        native.render(scene, [], {})
    except ValueError as error:
        assert "length" in str(error), str(error)
    else:
        raise AssertionError("invalid matrix accepted")
    scene["camera"]["transform"] = [1.0] * 16
    scene["camera"]["fov_y_rad"] = float("nan")
    try:
        native.render(scene, [], {})
    except ValueError as error:
        assert "non-finite" in str(error), str(error)
    else:
        raise AssertionError("non-finite value accepted")
    scene["camera"]["fov_y_rad"] = 1.0
    scene["materials"][0]["secondary"] = scene["materials"][0]
    try:
        native.render(scene, [], {})
    except ValueError as error:
        assert "nesting" in str(error), str(error)
    else:
        raise AssertionError("cyclic material accepted")


if "--scene" in sys.argv:
    scene_meshes()
else:
    mesh_fixtures()
    pixels()
    scene_description()
print("[native-scene-test] PASS", flush=True)
