"""Check whether bmw27's emissive meshes (Emit / HeadLightGlass /
HeadLightAngelEye) are registered as mesh-lights in the export, or fall into
the 'emission dropped on diffuse/glossy bounce' case."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "blender"))
for _m in list(sys.modules):
    if _m == "vibrt_blender" or _m.startswith("vibrt_blender."):
        del sys.modules[_m]
import bpy
from vibrt_blender import exporter

dg = bpy.context.evaluated_depsgraph_get()
json_str, _bin, _texs = exporter.export_scene_to_memory(dg, texture_pct=5)
scene = json.loads(json_str)

ml = scene.get("mesh_lights", [])
print(f"=== mesh_lights: {len(ml)} ===")
objs = scene.get("objects", [])
mats = scene.get("materials", [])
for m in ml:
    oi = m.get("object_idx")
    print(f"  mesh_light object_idx={oi} n_tris={len(m.get('triangles', m.get('tris', [])))}")

# Which materials have emission > 0 in the export, and are they mesh-lit?
print(f"\n=== materials with emission (of {len(mats)}) ===")
for i, mat in enumerate(mats):
    em = mat.get("emission", [0, 0, 0])
    if any(e > 1e-4 for e in em):
        print(f"  mat[{i}] emission={[round(e,3) for e in em]} emission_tex={'emission_tex' in mat} alpha_blend={mat.get('alpha_blend')}")

# Also list lights.json (rect promotions) if any.
print(f"\n=== lights: {len(scene.get('lights', []))} ===")
for L in scene.get("lights", []):
    print(f"  {L.get('type')} color={[round(x,3) for x in L.get('color',[])]} power={L.get('power')}")
print("=== DONE ===", flush=True)
