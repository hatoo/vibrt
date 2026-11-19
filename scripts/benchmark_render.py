"""Benchmark steady-state progressive rendering separately from preparation.

blender -b --factory-startup scene.blend --python scripts/benchmark_render.py -- \
    --output target/render-bench/current --percentage 25 --spp 32
Use --repo to compare an isolated checkout of an older renderer.
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys
import time

import bpy
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--percentage", type=int, default=25)
parser.add_argument("--spp", type=int, default=32)
parser.add_argument("--repeat", type=int, default=2)
parser.add_argument("--batch", action="store_true")
parser.add_argument("--without-graphs", action="store_true",
                    help="Diagnostic ablation, changes materials")
parser.add_argument("--without-bump", action="store_true",
                    help="Diagnostic ablation, changes materials")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
if args.spp < 1 or args.repeat < 1 or not 1 <= args.percentage <= 100:
    parser.error("spp and repeat must be positive; percentage must be 1..100")
sys.path.insert(0, str(args.repo.resolve() / "blender"))
import vibrt_blender
from vibrt_blender import exporter, vibrt_native as native

vibrt_blender.register()
scene = bpy.context.scene
scene.render.engine = "VIBRT"
scene.render.resolution_percentage = args.percentage
scene.vibrt_spp = args.spp
scene.vibrt_denoise = False
args.output.mkdir(parents=True, exist_ok=True)
export_started = time.perf_counter()
description, blobs, textures = exporter.export_scene_to_memory(
    bpy.context.evaluated_depsgraph_get(), serialize=False
)
export_seconds = time.perf_counter() - export_started
operations = {}
graph_count = 0
graph_lengths = []


def visit(material):
    global graph_count
    graph = material.get("shader_graph")
    if graph:
        graph_count += 1
        graph_lengths.append(len(graph["nodes"]))
        for node in graph["nodes"]:
            operations[node["op"]] = operations.get(node["op"], 0) + 1
        if args.without_graphs:
            material.pop("shader_graph")
        elif args.without_bump:
            for output in ("bump_height", "bump_strength", "bump_distance"):
                graph["outputs"].pop(output, None)
            if not graph["outputs"]:
                material.pop("shader_graph")
    for child in ("secondary", "left_subtree"):
        if material.get(child):
            visit(material[child])


for material in description["materials"]:
    visit(material)
(args.output / "scene.json").write_text(json.dumps(description, indent=2), encoding="utf-8")
report = {
    "blender": bpy.app.version_string,
    "repo": str(args.repo.resolve()),
    "export_seconds": export_seconds,
    "percentage": args.percentage,
    "spp": args.spp,
    "batch": args.batch,
    "graphs": graph_count,
    "graph_lengths": graph_lengths,
    "operations": operations,
    "without_graphs": args.without_graphs,
    "without_bump": args.without_bump,
    "runs": [],
}
byte_blobs = [b.view(np.uint8) for b in blobs]

for run in range(args.repeat):
    messages, samples = [], []
    start = time.perf_counter()

    def log(message):
        messages.append(message)
        print(message, flush=True)

    def preview(pixels, completed, total):
        samples.append((completed, time.perf_counter() - start))

    opts = {"spp": args.spp, "denoise": False, "preview_interval_ms": 0}
    pixels, _ = native.render(
        description, byte_blobs, opts, log_cb=log,
        texture_arrays=textures, preview_cb=None if args.batch else preview,
    )
    elapsed = time.perf_counter() - start
    per_sample = [(t2 - t1) / (s2 - s1)
                  for (s1, t1), (s2, t2) in zip(samples, samples[1:]) if s2 > s1]
    render_line = next(m for m in messages if m.startswith("Render: "))
    match = re.search(r"([0-9.]+)(ns|µs|μs|us|ms|s)", render_line)
    if match is None:
        raise RuntimeError(f"Unrecognized render duration: {render_line}")
    units = {"ns": 1e-9, "µs": 1e-6, "μs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1}
    render_seconds = float(match[1]) * units[match[2]]
    record = {
        "elapsed_seconds": elapsed,
        "render_seconds": render_seconds,
        "mean_sample_ms": render_seconds / args.spp * 1000,
        "median_sample_ms": statistics.median(per_sample) * 1000 if per_sample else None,
        "p90_sample_ms": float(np.percentile(per_sample, 90)) * 1000 if per_sample else None,
        "preview_samples": samples,
        "shape": list(pixels.shape),
        "mean_rgb": pixels[:, :, :3].mean(axis=(0, 1)).tolist(),
    }
    report["runs"].append(record)
    np.save(args.output / "pixels.npy", pixels)
    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[render-bench]", json.dumps({k: v for k, v in record.items()
                                       if k != "preview_samples"}), flush=True)
