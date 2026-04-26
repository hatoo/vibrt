"""Discover and invoke `vibrt_native` — the in-process PyO3 extension.

The addon renders exclusively through the bundled `vibrt_native.pyd`. The
standalone `vibrt.exe` binary still exists for CLI tooling but is not
invoked from the addon itself.
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
    scene_json: str,
    scene_bin: bytes,
    report,
    is_break,
    denoise: bool = False,
    texture_arrays=None,
):
    """Render `(scene.json, scene.bin)` in-process via `vibrt_native`.

    Returns a `(height, width, 4)` float32 numpy ndarray (linear RGBA, the
    same buffer the GPU writes — bottom-left origin matching Blender's
    `Image.pixels`). Raises `ImportError` if the extension isn't available;
    raises `RuntimeError` for vibrt errors; raises `KeyboardInterrupt` if
    the user aborted via Esc.

    `texture_arrays`, when supplied, is the per-texture pixel-array list
    produced by `exporter.export_scene_to_memory`. The Rust loader resolves
    each `TextureDesc.array_index` against it, so texture pixels can be
    handed across PyO3 directly instead of being concatenated into the bin.
    """
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

    opts = {"denoise": bool(denoise)}
    return native.render(
        scene_json, scene_bin, opts, log_cb, cancel_cb,
        texture_arrays=texture_arrays,
    )
