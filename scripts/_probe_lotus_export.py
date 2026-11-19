"""Run the vibrt material exporter on lotus_leafs and dump the resulting
PrincipledGpu params + color_graph so we can see what the GPU actually gets."""
import sys, os, json
sys.path.insert(0, os.path.abspath("blender"))

import bpy
import vibrt_blender.material_export as me

mat = bpy.data.materials.get("white_lotus_leafs")
print(f"=== material: {mat.name} ===")

# These are kwargs the real exporter passes
class FakeWriter:
    def __init__(self):
        self.bin = bytearray()
    def write_blob(self, b):
        off = len(self.bin)
        self.bin.extend(b)
        return {"offset": off, "len": len(b)}

textures = []
writer = FakeWriter()
me.begin_export()
params = me.export_material(mat, writer, textures)
me.end_export()
# Trim color_graph for printing
import pprint
def trim(d):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if k.endswith("_tex") and isinstance(v, dict):
                out[k] = {kk: vv for kk, vv in v.items() if kk != "blob"}
            elif k == "color_graph":
                out[k] = "<graph>"
            else:
                out[k] = trim(v)
        return out
    if isinstance(d, list):
        return [trim(x) for x in d]
    return d
print(json.dumps(trim(params), indent=2, default=str))
print(f"\ngraph nodes: {params.get('color_graph', {}).get('nodes', 'none')[:5] if 'color_graph' in params else 'no graph'}")
