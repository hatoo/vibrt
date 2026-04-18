"""Build vibrt_blender.zip for Blender installation.

Run: python blender/build_addon.py
"""

import zipfile
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "vibrt_blender"
OUT = HERE / "vibrt_blender.zip"


def main():
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in SRC.rglob("*"):
            if f.is_dir() or "__pycache__" in f.parts or f.suffix == ".pyc":
                continue
            z.write(f, f.relative_to(HERE))
    print(f"Wrote {OUT.relative_to(HERE.parent)} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
