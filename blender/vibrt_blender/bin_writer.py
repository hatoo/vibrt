"""Per-blob writer for the in-process FFI export path.

`write_f32` / `write_u32` append a contiguous numpy array to
`mesh_blobs` and return a plain integer index. The Rust loader reinterprets
each blob as f32 or u32 according to the scene field. `write_texture_pixels`
stores f32 RGBA data in a separate `texture_arrays` list and returns
`{"array_index": i}` for the texture descriptor. The writer retains these
arrays so the native binding can borrow them for the duration of rendering.

This module has no bpy imports, so tests can import and exercise it
under plain CPython (see `test_bin_writer.py`).
"""

from __future__ import annotations

import numpy as np


class BinWriter:
    __slots__ = ("_blobs", "_texture_arrays")

    def __init__(self):
        self._blobs: list = []
        self._texture_arrays: list = []

    @property
    def mesh_blobs(self) -> list:
        """Mesh / index / vertex-color / colour-graph LUT numpy arrays.
        Index matches the `array_index` returned by `write_f32` / `write_u32`."""
        return self._blobs

    @property
    def texture_arrays(self) -> list:
        """f32 RGBA numpy arrays accumulated by `write_texture_pixels`."""
        return self._texture_arrays

    def write_f32(self, xs) -> int:
        """Append a contiguous float32 blob and return its array_index.

        Mesh / index / vertex-color fields on `MeshDesc` are plain `u32`
        on the Rust side, so the index is returned bare — JSON ends up
        with `"vertices": 5` not `"vertices": {"array_index": 5}`."""
        arr = xs if isinstance(xs, np.ndarray) else np.asarray(xs, dtype=np.float32)
        if arr.dtype != np.float32 or not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        i = len(self._blobs)
        self._blobs.append(arr)
        return i

    def write_u32(self, xs) -> int:
        """Append a contiguous uint32 blob and return its array_index."""
        arr = xs if isinstance(xs, np.ndarray) else np.asarray(xs, dtype=np.uint32)
        if arr.dtype != np.uint32 or not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=np.uint32)
        i = len(self._blobs)
        self._blobs.append(arr)
        return i

    def write_texture_pixels(self, arr) -> dict:
        """Park a contiguous float32 RGBA buffer into the per-texture array
        list. The texture data crosses PyO3 as a typed `PyBuffer<f32>` (no
        u8 view trickery) so it's bagged separately from mesh blobs.
        """
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr, dtype=np.float32)
        if arr.dtype != np.float32 or not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        i = len(self._texture_arrays)
        self._texture_arrays.append(arr)
        return {"array_index": i}
