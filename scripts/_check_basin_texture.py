"""Verify that vibrt's `pebbles.ground` material is correctly using
rocks.jpg by walking the exporter output."""
import bpy, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
from vibrt_blender import material_export
from vibrt_blender.export_context import ExportContext
export_context = ExportContext()


class FakeWriter:
    def __init__(self):
        self.bin = bytearray()
        self.array_idx = 0

    def write_blob(self, b):
        off = len(self.bin)
        self.bin.extend(b)
        return {"offset": off, "len": len(b)}

    def write_texture_pixels(self, pixels):
        i = self.array_idx
        self.array_idx += 1
        return {"array_index": i}


mat = bpy.data.materials.get("pebbles.ground")
print(f"material: {mat.name}")

writer = FakeWriter()
textures = []
material_export.begin_export(context=export_context)
params = material_export.export_material(mat, writer, textures, context=export_context)
material_export.end_export(context=export_context)

print(f"\nExported params:")
for k, v in params.items():
    if "tex" in k or "graph" in k:
        if isinstance(v, dict):
            print(f"  {k}: {dict((kk, vv) for kk, vv in v.items() if kk != 'pixels')}")
        else:
            print(f"  {k}: {v}")
    else:
        print(f"  {k}: {v}")

print(f"\nTextures exported ({len(textures)}):")
for i, t in enumerate(textures):
    print(f"  [{i}] {t.get('_key', '?')} {t.get('width')}x{t.get('height')} colorspace={t.get('colorspace')}")
