"""Exercise the asynchronous F12 render path, which background renders miss.

Run: blender scene.blend --python scripts/test_render_job.py -- --expect-sky
The test saves previews under target/render-job-test and exits Blender.
It does not save the scene or preferences. Use a separate Blender process.
"""

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

import addon_utils
import bpy
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("target/render-job-test"))
parser.add_argument("--percentage", type=int, default=25)
parser.add_argument("--spp", type=int, default=4)
parser.add_argument("--repeat", type=int, default=2)
parser.add_argument("--expect-sky", action="store_true")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
assert not bpy.app.background, "Use a normal Blender process to test the render job thread"
assert args.repeat > 0
args.output.mkdir(parents=True, exist_ok=True)
if not addon_utils.check("vibrt_blender")[1]:
    addon_utils.enable("vibrt_blender")
from vibrt_blender import exporter

scene = bpy.context.scene
scene.render.engine = "VIBRT"
scene.render.resolution_percentage = args.percentage
scene.render.image_settings.file_format = "PNG"
scene.vibrt_spp = args.spp
scene.vibrt_denoise = False
bpy.context.preferences.use_preferences_save = False
bpy.context.preferences.view.render_display_type = "NONE"
state = {"count": 0, "finished": False, "cancelled": False, "started": 0.0}


def scene_ids():
    return tuple(tuple(item.as_pointer() for item in getattr(bpy.data, name))
                 for name in ("scenes", "objects", "cameras", "worlds", "materials"))


original_ids = scene_ids()


def completed(*_args):
    state["finished"] = True


def cancelled(*_args):
    state["cancelled"] = True


def begin_render():
    state["finished"] = False
    state["started"] = time.monotonic()
    scene.render.filepath = str((args.output / f"render-{state['count'] + 1}.png").resolve())
    result = bpy.ops.render.render("INVOKE_DEFAULT", write_still=True)
    assert "RUNNING_MODAL" in result, result
    print("[render-job-test] started", state["count"] + 1, flush=True)


def watch():
    try:
        if not state["started"]:
            begin_render()
            return 0.2
        assert time.monotonic() - state["started"] < 180, "Render timed out"
        assert not state["cancelled"], "Render was cancelled"
        if not state["finished"] or bpy.app.is_job_running("RENDER"):
            return 0.2
        output = Path(scene.render.filepath)
        assert output.is_file(), f"No render output: {output}"
        image = bpy.data.images.load(str(output), check_existing=False)
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(-1, 4)
        assert np.isfinite(pixels).all() and float(np.ptp(pixels[:, :3])) > 0.01
        bpy.data.images.remove(image)
        assert scene_ids() == original_ids, "Render changed the user's scene datablocks"
        if args.expect_sky:
            assert exporter._SKY_BAKE_CACHE, "Sky bake was skipped"
        state["count"] += 1
        print("[render-job-test] PASS", state["count"], str(output), flush=True)
        if state["count"] < args.repeat:
            begin_render()
            return 0.2
        (args.output / "success.txt").write_text(
            f"{state['count']} asynchronous renders passed\n", encoding="utf-8"
        )
        # The job has finished and files are closed. A normal Blender quit
        # would overwrite the user's Recover Last Session (quit.blend).
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


bpy.app.handlers.render_complete.append(completed)
bpy.app.handlers.render_cancel.append(cancelled)
bpy.app.timers.register(watch, first_interval=1.0)
