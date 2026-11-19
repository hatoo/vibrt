"""Pinpoint exactly which NewGeometry output drives dayLight_portal's
MixShader factor, then verify what _from_mix actually does at export.
"""
import bpy

mat = bpy.data.materials["dayLight_portal"]
out = next(n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL")
mix = out.inputs["Surface"].links[0].from_node
fac_link = mix.inputs[0].links[0]
print(f"Mix.Factor ← {fac_link.from_node.bl_idname}/{fac_link.from_node.name} "
      f"output socket = {fac_link.from_socket.name!r}")
print(f"Mix.Shader (1) = {mix.inputs[1].links[0].from_node.bl_idname}")
print(f"Mix.Shader (2) = {mix.inputs[2].links[0].from_node.bl_idname}")

# Now exercise the exporter against this material to see what the
# resulting dict looks like.
import sys
HERE = "c:\\Users\\hato2\\Desktop\\vibrt"
sys.path.insert(0, HERE + "\\blender")
from vibrt_blender import material_export as me

# Mock writer + textures.
class FakeWriter:
    def add_blob(self, *a, **kw): return 0
    def write_texture_pixels(self, *a, **kw): return 0

textures = []
result = me.export_material(mat, FakeWriter(), textures)
print(f"\n=== exporter returned ===")
import json
def safe(o):
    if isinstance(o, dict): return {k: safe(v) for k, v in o.items()}
    if isinstance(o, list): return [safe(x) for x in o]
    if isinstance(o, (int, float, str, bool)) or o is None: return o
    return repr(o)
print(json.dumps(safe(result), indent=2)[:2000])
