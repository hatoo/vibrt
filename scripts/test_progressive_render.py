"""GPU regression for progressive accumulation, denoising, and cancellation.

blender -b --factory-startup --python-exit-code 1 --python scripts/test_progressive_render.py
Optional -- --save-baseline records the old extension's output before rebuilding.
"""

from pathlib import Path
import sys
import time

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
many_opts = dict(opts, spp=257, preview_interval_ms=100)
many, many_depth = render(many_opts)
if "--save-baseline" in sys.argv:
    np.savez(output / "baseline.npz", pixels=reference, depth=reference_depth,
             denoised=denoised, denoised_depth=denoised_depth,
             many=many, many_depth=many_depth)
    print("[progressive-test] saved pre-change baseline", flush=True)
    raise SystemExit(0)

baseline = output / "baseline.npz"
if baseline.exists():
    with np.load(baseline) as old:
        np.testing.assert_array_equal(reference, old["pixels"])
        np.testing.assert_array_equal(reference_depth, old["depth"])
        np.testing.assert_array_equal(denoised, old["denoised"])
        np.testing.assert_array_equal(denoised_depth, old["denoised_depth"])
        if "many" in old:
            np.testing.assert_array_equal(many, old["many"])
            np.testing.assert_array_equal(many_depth, old["many_depth"])
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

# Reuse full batches, then finish a partial batch.
# Retained snapshots must stay unchanged while the next batch runs on the GPU.
batched_frames = []


def slow_preview(pixels, samples, total):
    assert total == many_opts["spp"]
    saved = pixels.copy()
    time.sleep(0.01)
    np.testing.assert_array_equal(pixels, saved)
    batched_frames.append((samples, pixels))


pixels, depth = render(many_opts, preview_cb=slow_preview)
completed = [samples for samples, _ in batched_frames]
assert completed[0] == 1 and completed[-1] == many_opts["spp"]
assert completed == sorted(set(completed))
assert len(completed) < many_opts["spp"], "fast samples should skip intermediate displays"
np.testing.assert_array_equal(pixels, many)
np.testing.assert_array_equal(depth, many_depth)
for samples, snapshot in batched_frames:
    expected, _ = render(dict(many_opts, spp=samples))
    np.testing.assert_array_equal(snapshot, expected)

frames.clear()
pixels, depth = render(dict(opts, denoise=True, preview_interval_ms=60000), preview_cb=preview)
np.testing.assert_array_equal(pixels, denoised)
np.testing.assert_array_equal(depth, denoised_depth)
np.testing.assert_array_equal(frames[-1][1], reference)

frames.clear()
try:
    render(preview_cb=preview, cancel_cb=lambda: bool(frames and frames[-1][0] >= 3))
except KeyboardInterrupt:
    pass
else:
    raise AssertionError("cancellation was ignored")
assert [sample for sample, _ in frames] == [1, 2, 3]

# Cancellation can happen inside a display callback with GPU work queued ahead.
frames.clear()
try:
    render(dict(opts, preview_interval_ms=100), preview_cb=preview,
           cancel_cb=lambda: bool(frames))
except KeyboardInterrupt:
    pass
else:
    raise AssertionError("batched cancellation was ignored")
assert [sample for sample, _ in frames] == [1]


class CallbackFailure(Exception):
    pass


def fail(*args):
    raise CallbackFailure("preview failed")


for interval in (0, 100):
    try:
        render(dict(opts, preview_interval_ms=interval), preview_cb=fail)
    except CallbackFailure as error:
        assert str(error) == "preview failed"
    else:
        raise AssertionError("callback error was swallowed")

# Reach steady full batches before cancelling/failing a display callback. A
# large sample budget guarantees another batch is in flight at that point.
for callback_error in (False, True):
    late_samples = []

    def stop_late(pixels, samples, total):
        late_samples.append(samples)
        if callback_error and samples > 1:
            raise CallbackFailure("late preview failed")

    try:
        render(dict(many_opts, spp=1000000), preview_cb=stop_late,
               cancel_cb=lambda: len(late_samples) > 1)
    except (KeyboardInterrupt, CallbackFailure) as error:
        assert isinstance(error, CallbackFailure if callback_error else KeyboardInterrupt)
    else:
        raise AssertionError("queued batch cancellation/error was ignored")
    assert late_samples[0] == 1 and 1 < late_samples[-1] < 1000000

# Error/cancel paths must drain pending work and leave subsequent renders usable.
pixels, depth = render(many_opts)
np.testing.assert_array_equal(pixels, many)
np.testing.assert_array_equal(depth, many_depth)

for options in (dict(opts, spp=0), dict(opts, width=0), dict(opts, height=0)):
    try:
        render(options)
    except RuntimeError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid render dimensions/sample count accepted")

# Named AOVs accumulate separately from the display image. Exercise varying
# procedural values over enough samples to reuse the launch-parameter slots.
aov = bpy.context.view_layer.aovs.add()
aov.name = "Progressive Noise"
aov.type = "COLOR"
tree = bpy.data.objects["Cube"].active_material.node_tree
noise = tree.nodes.new("ShaderNodeTexNoise")
aov_node = tree.nodes.new("ShaderNodeOutputAOV")
aov_node.aov_name = aov.name
tree.links.new(noise.outputs["Color"], aov_node.inputs["Color"])
scene, blobs, textures = exporter.export_scene_to_memory(
    bpy.context.evaluated_depsgraph_get(), serialize=False
)
blobs = [array.view(np.uint8) for array in blobs]
serial_aovs = {}
serial_pixels, serial_depth = render(
    dict(many_opts, preview_interval_ms=0), preview_cb=lambda *_: None,
    aov_outputs=serial_aovs,
)
assert np.ptp(serial_aovs[aov.name][:, :, :3]) > 0.01
for callback in (None, lambda *_: None):
    batched_aovs = {}
    pixels, depth = render(many_opts, preview_cb=callback, aov_outputs=batched_aovs)
    np.testing.assert_array_equal(pixels, serial_pixels)
    np.testing.assert_array_equal(depth, serial_depth)
    np.testing.assert_array_equal(batched_aovs[aov.name], serial_aovs[aov.name])
print("[progressive-test] PASS: exact samples, batching, denoise, snapshots, AOVs, throttling, cancellation, errors", flush=True)
