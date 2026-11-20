"""Geometry displacement through the same C++ graph evaluator used by CUDA."""

import numpy as np
from .shader_graph import Compiler, UnsupportedNode


def _validate_graph(graph):
    # Displacement runs before intersections exist. Never silently substitute
    # zero for inputs that require rays, instances or surface differentials.
    for node in graph["nodes"]:
        op, modes = node["op"], node["modes"]
        unsupported = op in {
            "fresnel",
            "layer_weight",
            "camera",
            "light_falloff",
            "tangent",
            "wireframe",
            "ambient_occlusion",
            "bevel",
            "raycast",
            "hair_tangent",
            "shifted_context",
            "bump",
            "vector_bump",
        }
        unsupported |= op == "context" and modes[0] not in {0, 1, 2, 3, 4, 5, 13}
        unsupported |= op == "vector_transform" and 2 in modes[1:3]
        unsupported |= op == "normal_map" and modes[0] == 0
        unsupported |= op == "attribute" and (
            node["attribute"].split(":", 1)[0] not in {"uv", "color", "attr", "tangent"}
            or modes[0] not in {0, 1, 2}
        )
        if unsupported:
            raise UnsupportedNode(
                f"Geometric displacement cannot evaluate {op} with these inputs"
            )


def apply(mesh, obj, arrays, writer, textures, attribute_loader):
    targets = []
    for index, slot in enumerate(obj.material_slots):
        mat = slot.material
        if (
            mat is None
            or not mat.use_nodes
            or getattr(mat, "displacement_method", "BUMP") == "BUMP"
        ):
            continue
        output = next(
            (
                n
                for n in mat.node_tree.nodes
                if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output
            ),
            None,
        )
        socket = output.inputs.get("Displacement") if output else None
        if socket is not None and socket.is_linked:
            from . import material_export

            compiler = Compiler(writer, textures)
            previous_material = material_export._CURRENT_MATERIAL
            try:
                material_export._CURRENT_MATERIAL = mat.name
                graph = compiler.graph({"displacement_vector": compiler.input(socket)})
                _validate_graph(graph)
            except UnsupportedNode as exc:
                raise UnsupportedNode(f"{obj.name}/{mat.name}: {exc}") from exc
            finally:
                material_export._CURRENT_MATERIAL = previous_material
            targets.append((index, graph))
    if not targets:
        return False
    from . import vibrt_native

    # Graph constants, coordinates and texture samples are evaluated in C++.
    # Python only prepares RNA buffers and scatters results to shared vertices.
    vertices = arrays["vertices"].reshape(-1, 3)
    normals = arrays["normals"].reshape(-1, 3)
    matrix = np.asarray(obj.matrix_world, dtype=np.float32)
    inverse = np.linalg.inv(matrix).astype(np.float32)
    world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    world_normals = normals @ inverse[:3, :3]
    world_normals /= np.maximum(
        np.linalg.norm(world_normals, axis=1, keepdims=True), 1e-20
    )
    triangles = world.reshape(-1, 3, 3)
    geometric_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    geometric_normals /= np.maximum(
        np.linalg.norm(geometric_normals, axis=1, keepdims=True), 1e-20
    )
    geometric_normals = np.repeat(geometric_normals, 3, axis=0)
    generated = (vertices - vertices.min(axis=0)) / np.maximum(
        np.ptp(vertices, axis=0), 1e-20
    )
    triangle_material = arrays.get(
        "material_indices", np.zeros(len(vertices) // 3, np.uint32)
    )
    delta = np.zeros_like(vertices)
    for material_index, graph in targets:
        selected = np.flatnonzero(np.repeat(triangle_material == material_index, 3))
        if not len(selected):
            continue
        named = {n["attribute"] for n in graph["nodes"] if n["op"] == "attribute"}
        attributes = attribute_loader(mesh, named, writer)
        occupied = {n["modes"][0] for n in graph["nodes"] if n["op"] == "context"}
        free = iter(
            i for i in (31, 21, 20, 19, 15, 14, 11, 10, 9, 8) if i not in occupied
        )
        bindings = []
        bound = {}
        for node in graph["nodes"]:
            if node["op"] != "attribute":
                continue
            key = (node["attribute"], node["modes"][0])
            if key in bound:
                node.update(op="context", modes=[bound[key], 0, 0, 0])
                continue
            context = next(free, None)
            if context is None:
                raise UnsupportedNode(
                    "Displacement uses too many independent attributes"
                )
            name = node["attribute"]
            data = (
                writer.mesh_blobs[attributes[name]].reshape(-1, 4)
                if name in attributes
                else np.zeros((len(vertices), 4), np.float32)
            )
            mode = node["modes"][0]
            if mode == 1:
                data = np.repeat(data[:, :3].mean(axis=1, keepdims=True), 4, axis=1)
            elif mode == 2:
                data = np.repeat(data[:, 3:4], 4, axis=1)
            elif mode:
                raise UnsupportedNode(
                    "This displacement attribute requires surface services"
                )
            bindings.append((context, data))
            bound[key] = context
            node.update(op="context", modes=[context, 0, 0, 0])
        texture_views = []
        for texture in textures:
            data = writer.texture_arrays[texture["array_index"]]
            width, height = texture["width"], texture["height"]
            data = data.reshape(height, width, -1)
            if data.shape[2] == 3:
                data = np.concatenate(
                    (data, np.ones((height, width, 1), np.float32)), axis=2
                )
            texture_views.append((np.ascontiguousarray(data), width, height))
        for start in range(0, len(selected), 8192):
            ids = selected[start : start + 8192]
            ctx = np.zeros((len(ids), 32, 4), np.float32)
            ctx[:, 0, :2] = arrays.get(
                "uvs", np.zeros(len(vertices) * 2, np.float32)
            ).reshape(-1, 2)[ids]
            ctx[:, 1, :3] = generated[ids]
            ctx[:, 2, :3] = vertices[ids]
            ctx[:, 3, :3] = world[ids]
            ctx[:, 4, :3] = world_normals[ids]
            ctx[:, 5, :3] = geometric_normals[ids]
            ctx[:, 13, :3] = normals[ids]
            ctx[:, 22:25, :] = matrix[:3, :]
            ctx[:, 25:28, :] = inverse[:3, :]
            for context, data in bindings:
                ctx[:, context, :] = data[ids]
            result = vibrt_native.shader_pixels(
                graph, ctx.reshape(-1, 128), texture_views
            )
            delta[ids] = result[:, 58, :3] @ inverse[:3, :3].T
    # A vertex shared across UV seams receives the mean of its corner samples,
    # preventing cracks between triangles (as in Cycles displacement averaging).
    corners = np.empty(len(vertices), np.int32)
    mesh.loop_triangles.foreach_get("vertices", corners)
    offsets = np.zeros((len(mesh.vertices), 3), np.float32)
    np.add.at(offsets, corners, delta)
    counts = np.bincount(corners, minlength=len(mesh.vertices))
    offsets /= np.maximum(counts[:, None], 1)
    positions = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", positions)
    mesh.vertices.foreach_set("co", (positions.reshape(-1, 3) + offsets).reshape(-1))
    mesh.update()
    mesh.calc_loop_triangles()
    return True
