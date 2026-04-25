"""Standalone smoke test: load vibrt_native, render cornell, compare to CLI.

Run after `cargo build --release --features python -p vibrt`. Usage:

    py scripts/_smoke_native.py

Reads test_scenes/cornell/scene.json + scene.bin from disk (so we're testing
just the FFI bridge, not the in-memory exporter wiring), runs vibrt_native.render,
saves a PNG, and prints whether it matches a CLI render of the same scene.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DLL = REPO / "target" / "release" / "vibrt_native.dll"
EXE = REPO / "target" / "release" / "vibrt.exe"
SCENE = REPO / "test_scenes" / "cornell"


def _import_native():
    if not DLL.exists():
        sys.exit(f"build vibrt_native.dll first (cargo build --features python): {DLL}")
    work = Path(tempfile.mkdtemp(prefix="vibrt_smoke_"))
    pyd = work / "vibrt_native.pyd"
    pyd.write_bytes(DLL.read_bytes())
    sys.path.insert(0, str(work))
    import vibrt_native
    return vibrt_native


def _save_png_from_rgba(arr: np.ndarray, path: Path) -> None:
    # vibrt's image_io::save_image does sRGB conversion + tonemap clamp; this
    # smoke path uses raw float→u8 only, so the PNG will differ from the CLI
    # output's encoding pipeline. We still hash the underlying *float* pixel
    # buffer for the equivalence check; this PNG is just a quick visual.
    h, w, c = arr.shape
    assert c == 4
    rgb = arr[..., :3]
    # Apply same sRGB-from-linear that image_io does.
    a = np.clip(rgb, 0.0, None)
    out = np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(a, 1.0 / 2.4) - 0.055)
    out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(out, mode="RGB").save(path)


def _hash_floats(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()[:16]


def main() -> int:
    json_text = (SCENE / "scene.json").read_text()
    bin_bytes = (SCENE / "scene.bin").read_bytes()
    print(f"scene.json: {len(json_text):,} chars  scene.bin: {len(bin_bytes):,} bytes")

    native = _import_native()

    logs: list[str] = []
    pixels = native.render(
        json_text,
        bin_bytes,
        {"spp": 64, "width": 256, "height": 256},
        lambda msg: logs.append(msg),
        None,
    )
    print(f"render returned ndarray shape={pixels.shape} dtype={pixels.dtype}")
    for ln in logs:
        print(f"  log: {ln}")

    out_native = SCENE / "_smoke_native.png"
    _save_png_from_rgba(np.asarray(pixels), out_native)
    print(f"saved {out_native}")

    # FFI determinism: a second call with the same inputs must produce
    # identical pixels. This catches uninitialised state or rng-seed leaks
    # across calls.
    pixels2 = native.render(
        json_text,
        bin_bytes,
        {"spp": 64, "width": 256, "height": 256},
        None, None,
    )
    h1 = _hash_floats(pixels)
    h2 = _hash_floats(pixels2)
    print(f"call1 hash: {h1}")
    print(f"call2 hash: {h2}")
    if h1 != h2:
        print("FAIL: two FFI calls produced different pixels")
        return 1

    # CLI parity: render the same scene to .raw (linear float, bottom-up —
    # same convention as the FFI-returned ndarray). Direct float equality is
    # the strongest possible check: identical kernels, identical RNG seeding,
    # identical scene-loading semantics.
    cli_raw = SCENE / "_smoke_cli.raw"
    subprocess.run(
        [str(EXE), str(SCENE / "scene.json"),
         "--output", str(cli_raw),
         "--spp", "64", "--width", "256", "--height", "256"],
        check=True, capture_output=True,
    )
    raw = cli_raw.read_bytes()
    assert raw[:4] == b"VBLT", "expected VBLT raw header from vibrt"
    import struct
    fw, fh, fc = struct.unpack("<III", raw[4:16])
    cli_pixels = np.frombuffer(raw[16:], dtype=np.float32).reshape(fh, fw, fc)
    print(f"CLI .raw shape={cli_pixels.shape}")
    if cli_pixels.shape != pixels.shape:
        print(f"FAIL: shape mismatch CLI={cli_pixels.shape} FFI={pixels.shape}")
        return 1
    h_cli = _hash_floats(cli_pixels)
    h_ffi = _hash_floats(pixels)
    print(f"CLI .raw hash:  {h_cli}")
    print(f"FFI ndarr hash: {h_ffi}")
    if h_cli == h_ffi:
        print("PASS: FFI and CLI produce bit-identical pixels")
        return 0
    diff = np.abs(cli_pixels - pixels)
    print(f"FAIL: max-abs-diff={float(diff.max()):.6f} mean={float(diff.mean()):.6f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
