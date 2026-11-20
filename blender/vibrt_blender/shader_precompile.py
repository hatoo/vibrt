"""Isolated shader-cache warmup; only the worker enters Blender's render job."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

active_job = None
last_message = ""


class CompileJob:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vibrt_shader_compile_")
        self.directory = Path(self.temporary.name)
        self.process = None
        self.output = None
        self.cancelled = False
        self.started = time.monotonic()

    def launch(self, command):
        self.output = (self.directory / "worker.log").open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        self.process = subprocess.Popen(
            command,
            stdout=self.output,
            stderr=subprocess.STDOUT,
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    def status(self):
        try:
            return json.loads(
                (self.directory / "status.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return "Starting shader compiler..."

    def done(self):
        return self.process is not None and self.process.poll() is not None

    def result(self):
        if not self.done():
            raise RuntimeError("Shader compilation is still running")
        result = self.directory / "result.json"
        if self.process.returncode != 0 or not result.exists():
            transcript = (self.directory / "worker.log").read_text(
                encoding="utf-8", errors="replace"
            )
            raise RuntimeError(f"Shader compiler failed:\n{transcript[-4000:]}")
        return json.loads(result.read_text(encoding="utf-8"))

    def cancel(self):
        self.cancelled = True
        self.close()

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.output is not None:
            self.output.close()
        self.temporary.cleanup()


def start(scene, view_layer):
    import bpy
    from . import runner

    global active_job, last_message
    if active_job is not None:
        raise RuntimeError("Shader compilation is already running")
    native = runner.find_native_module()
    if native is None or not hasattr(native, "precompile_shaders"):
        raise RuntimeError("Rebuild the vibrt native extension and restart Blender")
    job = CompileJob()
    try:
        # Writing a library preserves unsaved edits without changing the user's
        # filepath, save state, current scene, or Render Result.
        bpy.data.libraries.write(
            str(job.directory / "inputs.blend"),
            {scene},
            path_remap="ABSOLUTE",
            fake_user=True,
        )
        request = {"scene": scene.name, "view_layer": view_layer.name}
        (job.directory / "request.json").write_text(
            json.dumps(request), encoding="utf-8"
        )
        job.launch(
            [
                bpy.app.binary_path,
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python-exit-code",
                "1",
                "--python",
                str(Path(__file__).resolve()),
                "--",
                str(job.directory),
            ]
        )
    except BaseException:
        job.close()
        raise
    active_job = job
    last_message = "Starting shader compiler..."
    return job


def stop():
    global active_job, last_message
    if active_job is not None:
        active_job.cancel()
        active_job = None
        last_message = "Shader compilation cancelled"


atexit.register(stop)


def run_worker(directory):
    import bpy
    import vibrt_blender
    from . import exporter, runner
    from .engine import VibrtRenderEngine

    def status(message):
        path = directory / "status.tmp"
        path.write_text(json.dumps(message), encoding="utf-8")
        path.replace(directory / "status.json")
        print(f"[vibrt] {message}", flush=True)

    status("Loading scene snapshot...")
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vibrt_blender.register()
    with bpy.data.libraries.load(str(directory / "inputs.blend"), link=False) as (
        _,
        data,
    ):
        data.scenes = [request["scene"]]
    scene = data.scenes[0]
    if scene is None:
        raise RuntimeError("Could not load the shader compilation scene")
    bpy.context.window.scene = scene
    bpy.context.window.view_layer = scene.view_layers[request["view_layer"]]
    scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)
    errors, results = [], []

    class ShaderCompileEngine(VibrtRenderEngine):
        bl_idname = "VIBRT_PRECOMPILE"
        bl_label = "vibrt Shader Compilation"

        def render(self, depsgraph):
            try:
                status("Reading scene shaders...")
                description = exporter.export_shader_scene(depsgraph)
                stats = runner.find_native_module().precompile_shaders(
                    description, status
                )
                stats["depsgraph_mode"] = depsgraph.mode
                stats["materials"] = len(description["materials"])
                results.append(stats)
            except Exception as exc:
                errors.append(exc)

    bpy.utils.register_class(ShaderCompileEngine)
    scene.render.engine = ShaderCompileEngine.bl_idname
    # Blender needs a camera to build a RENDER depsgraph; no image is sampled.
    if scene.camera is None:
        camera = bpy.data.objects.new(
            "Compile Camera", bpy.data.cameras.new("Compile Camera")
        )
        scene.collection.objects.link(camera)
        scene.camera = camera
    scene.render.resolution_x = scene.render.resolution_y = 4
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    bpy.ops.render.render(scene=scene.name, layer=request["view_layer"])
    if errors:
        raise errors[0]
    if len(results) != 1:
        raise RuntimeError("Shader compiler did not complete the requested view layer")
    (directory / "result.json").write_text(json.dumps(results[0]), encoding="utf-8")
    status("Shaders ready")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vibrt_blender.shader_precompile import run_worker

    run_worker(Path(sys.argv[sys.argv.index("--") + 1]))
