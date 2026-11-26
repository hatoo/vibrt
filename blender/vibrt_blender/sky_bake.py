"""Prepare native sky maps, with an isolated Cycles fallback for unsupported graphs."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


def _needs_world_bake(world):
    from . import exporter

    if exporter._world_sky_node(world) is not None:
        return True
    if exporter._detect_mix_world_camera_ray_split(world) is not None:
        return True
    if exporter._detect_mix_world(world) is not None:
        return bool(list(exporter._walk_world_for_sky_nodes(world)))
    return exporter._world_needs_full_bake(world)


@contextmanager
def _start_worker(command, directory, cancel):
    """Start immediately; yield a waiter and reap on cancellation/export errors."""
    log_path = directory / "worker.log"
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        def wait():
            if cancel():
                raise KeyboardInterrupt("Sky preparation cancelled")
            while process.poll() is None:
                if cancel():
                    raise KeyboardInterrupt("Sky preparation cancelled")
                time.sleep(0.1)
            transcript = log_path.read_text(encoding="utf-8", errors="replace")
            if process.returncode:
                raise RuntimeError(
                    f"sky bake worker exited with {process.returncode}:\n{transcript[-4000:]}"
                )
            return transcript

        try:
            yield wait
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _run_worker(command, directory, cancel):
    with _start_worker(command, directory, cancel) as wait:
        return wait()


def prepare_scene(scene, *, cancel=lambda: False, log=print, width=None, height=None, context=None):
    """Blocking convenience wrapper for scripts and sky-only preparation."""
    from .export_context import ExportContext
    if context is None:
        context = ExportContext()
    with prepare_scene_async(scene, cancel=cancel, log=log, width=width, height=height, context=context) as wait:
        wait()
    return context


@contextmanager
def prepare_scene_async(scene, *, cancel=lambda: False, log=print, width=None, height=None, context):
    """Evaluate supported sky graphs natively; otherwise snapshot their inputs.

    No scenes, objects, node trees, or render results are created/edited in
    the caller. Blender's library writer captures the existing datablocks
    and their dependencies for the fallback; all scene construction and
    Cycles work happens in a fresh background Blender, with user addons
    and auto-execution off. Native evaluation creates only numeric buffers.

    Geometry and materials may be exported while the worker runs. Call the
    yielded function before exporting the world to publish the completed
    environment maps. Leaving the context always reaps the worker, even if
    export fails before reaching the world. No Python threads access bpy.
    """
    import bpy
    from . import exporter

    if (width is None) != (height is None):
        raise ValueError("Specify both sky bake dimensions or neither")
    from . import native_sky

    if native_sky.prepare_scene(scene, cancel=cancel, log=log, width=width, height=height, context=context):
        yield lambda: None
        return
    bake_world = _needs_world_bake(scene.world)
    portals = exporter._portal_emissions_to_bake(scene)
    if not bake_world and not portals:
        yield lambda: None
        return
    if cancel():
        raise KeyboardInterrupt("Sky preparation cancelled")
    # Preserve discovery order: the exporter uses the first portal's sky.
    materials = [mat for mat, _node in portals]
    datablocks = set(materials)
    if scene.world is not None:
        datablocks.add(scene.world)
    log("vibrt: preparing sky lighting in a background Blender")
    with tempfile.TemporaryDirectory(prefix="vibrt_sky_worker_") as temporary:
        directory = Path(temporary)
        bpy.data.libraries.write(
            str(directory / "inputs.blend"), datablocks,
            path_remap="ABSOLUTE", fake_user=True,
        )
        request = {
            "scene": scene.name,
            "world": scene.world.name if scene.world is not None else None,
            "materials": [mat.name for mat in materials],
            "frame": scene.frame_current,
            "subframe": scene.frame_subframe,
            "width": width, "height": height,
        }
        (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
        command = [
            bpy.app.binary_path, "--background", "--factory-startup",
            "--disable-autoexec", "--python-exit-code", "1",
            "--python", str(Path(__file__).resolve()), "--", str(directory),
        ]
        with _start_worker(command, directory, cancel) as wait:
            published = False

            def finish():
                nonlocal published
                if published:
                    return
                transcript = wait()
                for line in transcript.splitlines():
                    if line.startswith("[vibrt]"):
                        log(line)
                # Publish atomically only after every buffer is validated.
                context.sky_maps.update(_read_results(directory))
                published = True

            yield finish


def _read_results(directory):
    import numpy as np

    manifest = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if not manifest:
        raise RuntimeError("Sky preparation produced no environment maps")
    cache = {}
    for index, entry in enumerate(manifest):
        pixels = np.load(directory / f"bake-{index}.npy", allow_pickle=False)
        w, h = entry["width"], entry["height"]
        if pixels.shape != (h, w, 3) or pixels.dtype != np.float32:
            raise RuntimeError("Sky bake worker returned an invalid pixel buffer")
        cache[entry["key"]] = (pixels, w, h, entry["sun"])
    return cache


def run_worker(directory, *, context=None):
    from .export_context import ExportContext
    if context is None:
        context = ExportContext()
    import bpy
    import numpy as np
    from . import exporter

    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(str(directory / "inputs.blend"), link=False) as (_, data):
        data.worlds = [request["world"]] if request["world"] is not None else []
        data.materials = request["materials"]
    if any(item is None for item in [*data.worlds, *data.materials]):
        raise RuntimeError("Could not load sky inputs")
    world = data.worlds[0] if data.worlds else None
    bpy.context.scene.world = world
    bpy.context.scene.frame_set(request["frame"], subframe=request["subframe"])
    # Portal discovery needs only names and material slots, not scene geometry.
    scene = SimpleNamespace(
        name=request["scene"], world=world,
        objects=[SimpleNamespace(type="MESH", material_slots=[SimpleNamespace(material=m)])
                 for m in data.materials],
    )
    exporter.clear_sky_bake_cache(context=context)
    # Preserve the exporter's distinct defaults: worlds use 2048x1024,
    # while portal skies use 1024x512. Tests can override both explicitly.
    resolution = {} if request["width"] is None else {
        "w": request["width"], "h": request["height"],
    }
    exporter.prebake_sky_envmaps_for_world(world, **resolution, context=context)
    exporter.prebake_portal_emissions_for_scene(scene, **resolution, context=context)
    manifest = []
    for index, (key, (pixels, w, h, sun)) in enumerate(context.sky_maps.items()):
        np.save(directory / f"bake-{index}.npy", pixels, allow_pickle=False)
        manifest.append({"key": key, "width": w, "height": h, "sun": sun})
    (directory / "result.json").write_text(json.dumps(manifest), encoding="utf-8")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vibrt_blender.sky_bake import run_worker

    run_worker(Path(sys.argv[sys.argv.index("--") + 1]))
