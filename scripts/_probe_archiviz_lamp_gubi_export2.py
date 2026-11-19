"""Run vibrt's exporter on every material with emission != 0 and print the
material name + result, so we can match material indices in the JSON to
Blender material names — and confirm whether Lamp Gubi's strength=2 made it
through `_from_principled`."""
import bpy, sys, os
sys.path.insert(0, os.path.abspath("blender"))
import vibrt_blender.exporter as exporter
import vibrt_blender.material_export as material_export

class _DummyWriter:
    def __init__(self):
        self.buf = bytearray()
    def append(self, data):
        offset = len(self.buf)
        self.buf.extend(data)
        return (offset, len(data))

writer = _DummyWriter()
textures = []

interesting = ("Lamp Gubi", "Lampshade")
for m in bpy.data.materials:
    if m.name in interesting:
        try:
            params = material_export.export_material(m, writer, textures)
            print(f"=== {m.name!r} export result ===")
            for k, v in params.items():
                if k == "color_graph":
                    print(f"  {k}: <{len(v)} ops>")
                elif isinstance(v, dict):
                    print(f"  {k}: {{...{len(v)} keys...}}")
                else:
                    print(f"  {k}: {v}")
        except Exception as ex:
            print(f"!! {m.name!r}: {ex}")
        print()
