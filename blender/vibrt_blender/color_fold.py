"""Constant-fold color graphs using the shared C++/CUDA node evaluator.

Without a staged extension, leave the graph intact: folding is an optimization.
This module does not import bpy; the CPU backend is tested under plain Python.
"""

from . import node_ops


def fold_constants(graph: dict) -> dict:
    result = {"nodes": [dict(node) for node in graph.get("nodes", [])]}
    if graph.get("output") is not None:
        result["output"] = graph["output"]
    native = node_ops.native()
    if native is None:
        return result
    try:
        constants = native.fold_color_constants(graph)
    except (ValueError, TypeError):
        # The renderer's graph validator reports unsupported node schemas.
        return result
    for index, value in enumerate(constants):
        if value is not None:
            result["nodes"][index] = {"type": "const", "rgb": value}
    return result
