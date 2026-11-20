"""Discover and invoke `vibrt_native` — the in-process PyO3 extension.

The addon renders through the bundled `vibrt_native` extension. Command-line
helpers launch headless Blender and use this same in-process path.
"""

from __future__ import annotations


def find_native_module():
    """Import `vibrt_native` if it's bundled with the addon. Returns the
    module on success, or None on `ImportError` (extension not built yet,
    binary missing, ABI mismatch). Callers report a clear error and stop —
    there is no subprocess fallback.
    """
    try:
        from . import vibrt_native  # type: ignore  # ships as a .pyd next to __init__.py
        return vibrt_native
    except ImportError:
        return None


def run_render_inproc(
    scene_json: str | dict,
    mesh_blobs,
    report,
    is_break,
    denoise: bool = False,
    texture_arrays=None,
    preview_cb=None,
    aov_outputs=None,
):
    """Render a live scene dict (or legacy JSON) and borrowed pixel/mesh buffers.

    Returns a `(pixels, depth)` tuple where `pixels` is a
    `(height, width, 4)` float32 ndarray (linear RGBA, bottom-left origin)
    and `depth` is a `(height, width)` float32 ndarray of per-pixel
    primary-ray hit distance from the camera, in metres (misses store
    the camera's `clip_end`). The addon turns `depth` into Mist / Z
    passes for the compositor. Raises `ImportError` if the extension
    isn't available; raises `RuntimeError` for vibrt errors; raises
    `KeyboardInterrupt` if the user aborted via Esc.

    `mesh_blobs` is the per-blob byte-buffer list (mesh / index / vertex-
    color / colour-graph LUT data). Each entry is a numpy array; the Rust
    side indexes them via fields such as `MeshDesc.vertices` and `.indices`.
    `texture_arrays` is the parallel list for `TextureDesc.array_index`.
    `preview_cb(pixels, samples, total)` receives normalized, owned RGBA
    snapshots up to 10 times per second, plus the first/final sample.
    Both paths queue short batches of one-sample launches; omitting the callback
    disables intermediate readback. If supplied,
    `aov_outputs` receives the named final AOV arrays.
    """
    import numpy as np
    native = find_native_module()
    if native is None:
        raise ImportError("vibrt_native not available (build with --features python)")

    def log_cb(msg: str) -> None:
        # Filter out empty lines so the Info panel stays tidy. Strip CR
        # which Windows can introduce when stdout-style messages arrive.
        s = msg.rstrip()
        if s:
            report({"INFO"}, s)

    def cancel_cb() -> bool:
        # `is_break()` is the engine's `test_break` — flips when the user
        # hits Esc. Wrap it because PyO3 wants a plain truthiness check.
        try:
            return bool(is_break())
        except Exception:
            return False

    # Reinterpret each typed (f32 / u32) blob as a uint8 view so PyO3 can
    # take it as `PyBuffer<u8>`. `.view(np.uint8)` is a no-copy reinterpret.
    blobs_u8 = [b if b.dtype == np.uint8 else b.view(np.uint8) for b in mesh_blobs]

    opts = {"denoise": bool(denoise)}
    kwargs = {"texture_arrays": texture_arrays}
    if preview_cb is not None:
        kwargs["preview_cb"] = preview_cb
    if aov_outputs is not None:
        kwargs["aov_outputs"] = aov_outputs
    return native.render(
        scene_json, blobs_u8, opts, log_cb, cancel_cb,
        **kwargs,
    )
