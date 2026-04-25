"""Run the exporter twice — once into a file, once into a bytearray — and
diff the resulting bin bytes. Any difference points at a bug in BinWriter's
new bytearray-sink path.

Usage (inside Blender):
    blender -b scene.blend --python scripts/_diff_bins.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "blender"))
for mod_name in list(sys.modules):
    if mod_name == "vibrt_blender" or mod_name.startswith("vibrt_blender."):
        del sys.modules[mod_name]
from vibrt_blender import exporter  # noqa: E402


depsgraph = bpy.context.evaluated_depsgraph_get()

# File path (subprocess-style)
work = Path(tempfile.mkdtemp())
exporter.export_scene(depsgraph, work / "scene.json", work / "scene.bin")
file_bin = (work / "scene.bin").read_bytes()
print(f"file: {len(file_bin):,} bytes")

# In-memory path
json_str, mem_bin = exporter.export_scene_to_memory(depsgraph)
print(f"mem:  {len(mem_bin):,} bytes ({type(mem_bin).__name__})")

if len(file_bin) != len(mem_bin):
    print(f"FAIL: length mismatch ({len(file_bin)} vs {len(mem_bin)})")
    sys.exit(1)

# Compare in chunks to keep memory usage flat.
import hashlib
print("file sha256:", hashlib.sha256(file_bin).hexdigest()[:16])
print("mem  sha256:", hashlib.sha256(bytes(mem_bin)).hexdigest()[:16])

# Find first diff
mv_a = memoryview(file_bin)
mv_b = memoryview(mem_bin)
chunk = 1 << 20
diffs = 0
first_diff = None
for i in range(0, len(file_bin), chunk):
    a = mv_a[i:i+chunk]
    b = mv_b[i:i+chunk]
    if a != b:
        # narrow down
        for j in range(len(a)):
            if a[j] != b[j]:
                if first_diff is None:
                    first_diff = i + j
                diffs += 1
        if diffs > 0:
            print(f"first diff at offset {first_diff}, chunk diffs continue...")
            break
print(f"differing chunks scanned: diffs={diffs}, first_diff={first_diff}")
