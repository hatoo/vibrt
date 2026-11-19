"""GPU regression for progressive accumulation, denoising, and cancellation.

blender -b --factory-startup --python-exit-code 1 --python scripts/test_progressive_render.py
Optional -- --save-baseline records the old extension's output before rebuilding.
"""

from pathlib import Path
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
import vibrt_blender
from vibrt_blender import exporter, vibrt_native as native

vibrt_blender.register()
bpy.context.scene.render.engine = "VIBRT"
scene, blobs, textures = exporter.export_scene_to_memory(bpy.context.evaluated_depsgraph_get(), serialize=False)
blobs = [array.view(np.uint8) for array in blobs]
opts = {"spp": 8, "width": 96, "height": 54, "preview_interval_ms": 0}
output = ROOT / "target/progressive-test"
output.mkdir(parents=True, exist_ok=True)


def render(options=None, **callbacks):
    return native.render(scene, blobs, options or opts, texture_arrays=textures, **callbacks)


reference, reference_depth = render()
denoised, denoised_depth = render(dict(opts, denoise=True))
if "--save-baseline" in sys.argv:
    np.savez(output / "baseline.npz", pixels=reference, depth=reference_depth,
             denoised=denoised, denoised_depth=denoised_depth)
    print("[progressive-test] saved pre-change baseline", flush=True)
    raise SystemExit(0)

baseline = output / "baseline.npz"
if baseline.exists():
    with np.load(baseline) as old:
        np.testing.assert_array_equal(reference, old["pixels"])
        np.testing.assert_array_equal(reference_depth, old["depth"])
        np.testing.assert_array_equal(denoised, old["denoised"])
        print("[progressive-test] pre-change image/depth/denoise exact", flush=True)

frames = []


def preview(pixels, samples, total):
    assert pixels.shape == (54, 96, 4) and pixels.dtype == np.float32
    assert np.isfinite(pixels).all()
    assert total == 8
    frames.append((samples, pixels))  # retaining an old snapshot must be safe


pixels, depth = render(preview_cb=preview)
assert [sample for sample, _ in frames] == list(range(1, 9))
np.testing.assert_array_equal(pixels, reference)
np.testing.assert_array_equal(depth, reference_depth)
np.testing.assert_array_equal(frames[-1][1], reference)
first, _ = render(dict(opts, spp=1))
np.testing.assert_array_equal(frames[0][1], first)
assert not np.array_equal(frames[0][1], frames[-1][1]), "image did not accumulate samples"

frames.clear()
pixels, depth = render(dict(opts, denoise=True), preview_cb=preview)
np.testing.assert_array_equal(pixels, denoised)
np.testing.assert_array_equal(depth, denoised_depth)
np.testing.assert_array_equal(frames[-1][1], reference)  # denoise only final output

# Long throttle still emits both first and final samples.
frames.clear()
pixels, _ = render(dict(opts, preview_interval_ms=60000), preview_cb=preview)
assert [sample for sample, _ in frames] == [1, 8]
np.testing.assert_array_equal(pixels, reference)

frames.clear()
try:
    render(preview_cb=preview, cancel_cb=lambda: bool(frames and frames[-1][0] >= 3))
except KeyboardInterrupt:
    pass
else:
    raise AssertionError("cancellation was ignored")
assert [sample for sample, _ in frames] == [1, 2, 3]


class CallbackFailure(Exception):
    pass


def fail(*args):
    raise CallbackFailure("preview failed")


try:
    render(preview_cb=fail)
except CallbackFailure as error:
    assert str(error) == "preview failed"
else:
    raise AssertionError("callback error was swallowed")

for options in (dict(opts, spp=0), dict(opts, width=0), dict(opts, height=0)):
    try:
        render(options)
    except RuntimeError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid render dimensions/sample count accepted")
print("[progressive-test] PASS: exact samples, denoise, snapshots, throttling, cancellation, errors", flush=True)
