"""Smoke test for the in-process render path.

Run via:
    blender -b test.blend --python scripts/_blender_inproc_test.py -- --output result.png

Drives the addon's `engine.render` through the in-process branch (which now
imports `vibrt_native` and skips the subprocess+disk roundtrip), then dumps
the Combined pass to a PNG so we can compare against the CLI baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "blender"))

# Force-fresh import of the source addon (the dev-junctioned addon may also
# be on sys.path — evict any cached bytes).
for mod_name in list(sys.modules):
    if mod_name == "vibrt_blender" or mod_name.startswith("vibrt_blender."):
        del sys.modules[mod_name]

import vibrt_blender  # noqa: E402
from vibrt_blender import engine, runner  # noqa: E402

mod = runner.find_native_module()
if mod is None:
    print("[smoke] vibrt_native not importable — engine.render will use subprocess fallback")
else:
    print(f"[smoke] vibrt_native loaded from {mod.__file__}")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--spp", type=int, default=4)
    ap.add_argument("--percentage", type=int, default=25)
    args = ap.parse_args(argv)

    scene = bpy.context.scene
    scene.render.engine = "VIBRT"
    scene.render.resolution_percentage = args.percentage
    scene.vibrt_spp = args.spp
    scene.render.filepath = args.output
    scene.render.image_settings.file_format = "PNG"

    bpy.ops.render.render(write_still=True)
    print(f"[smoke] wrote {args.output}")


vibrt_blender.register()
try:
    main()
finally:
    vibrt_blender.unregister()
