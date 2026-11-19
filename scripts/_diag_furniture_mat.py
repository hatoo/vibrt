"""Dump junk_shop 'Furniture' material: Principled inputs + how vibrt exports
it (subsurface weight/radius/color, base color, roughness)."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "blender"))
for _m in list(sys.modules):
    if _m == "vibrt_blender" or _m.startswith("vibrt_blender."):
        del sys.modules[_m]
import bpy
from vibrt_blender import material_export as ME
from vibrt_blender.bin_writer import BinWriter

mat = bpy.data.materials.get("Furniture")
print("Furniture found:", mat is not None)
if mat and mat.use_nodes:
    for n in mat.node_tree.nodes:
        if n.bl_idname == "ShaderNodeBsdfPrincipled":
            print("=== Principled inputs (non-default / relevant) ===")
            for name in ("Base Color", "Metallic", "Roughness", "IOR",
                         "Subsurface Weight", "Subsurface Radius", "Subsurface Scale",
                         "Subsurface Anisotropy", "Transmission Weight", "Alpha",
                         "Specular IOR Level", "Sheen Weight", "Coat Weight",
                         "Emission Color", "Emission Strength"):
                inp = n.inputs.get(name)
                if inp is None:
                    continue
                if inp.is_linked:
                    src = inp.links[0].from_node
                    print(f"  {name}: LINKED <- {src.bl_idname} {src.name!r}")
                else:
                    try:
                        val = tuple(inp.default_value) if hasattr(inp.default_value, '__len__') else inp.default_value
                    except Exception:
                        val = inp.default_value
                    print(f"  {name}: {val}")

# Export the material and dump the vibrt params.
writer = BinWriter(); textures = []
try:
    params = ME.export_material(mat, writer, textures)
    print("\n=== vibrt exported params (subsurface/emission/base/roughness) ===")
    for k in sorted(params.keys()):
        if any(s in k for s in ("sss", "subsurface", "emission", "base_color", "roughness",
                                "transmission", "metallic", "sheen", "coat", "alpha", "secondary",
                                "translucent", "diffuse")):
            val = params[k]
            if isinstance(val, dict):
                val = f"<dict keys={list(val.keys())}>"
            print(f"  {k}: {val}")
except Exception as e:
    import traceback; traceback.print_exc()
print("=== DONE ===", flush=True)
