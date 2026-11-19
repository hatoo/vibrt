"""Identify which junk_shop materials drive the bright over-delivery.
Loads the vibrt/cycles EXRs, finds the worst-over bright pixels, ray_casts
each to its material, and aggregates the over-energy by material.
"""
import sys
from pathlib import Path
import numpy as np
import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent.parent
VE = HERE / "test_scenes" / "_eng_diff" / "junk_shop_vibrt.exr"
CE = HERE / "test_scenes" / "_eng_diff" / "junk_shop_cycles.exr"


def load(p):
    img = bpy.data.images.load(str(p))
    img.colorspace_settings.name = "Linear Rec.709"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    bpy.data.images.remove(img)
    return flat.reshape(h, w, 4)[..., :3][::-1].copy(), w, h


v, w, h = load(VE)
c, _, _ = load(CE)
vlum = v @ np.array([0.2126, 0.7152, 0.0722], np.float32)
clum = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
over = vlum - clum
# Focus on bright over-delivering pixels.
mask = (clum > 0.15) & (over > 0.15)
ys, xs = np.where(mask)
order = np.argsort(-over[ys, xs])
ys, xs = ys[order], xs[order]
print(f"bright over-delivering pixels: {len(ys)} (analyzing top 400)")

scene = bpy.context.scene
dg = bpy.context.evaluated_depsgraph_get()
cam = scene.camera
cam_mtx = cam.matrix_world
frame = [cam_mtx @ p for p in cam.data.view_frame(scene=scene)]
tr, br, bl, tl = frame
origin = cam_mtx.translation


def ray_dir(px, py):  # px,py in [0,1], py=0 top
    top = tl.lerp(tr, px); bot = bl.lerp(br, px)
    p = bot.lerp(top, 1 - py)
    return (p - origin).normalized()


from collections import defaultdict
mat_over = defaultdict(float)
mat_cnt = defaultdict(int)
mat_emissive = {}
N = min(400, len(ys))
for i in range(N):
    yy, xx = int(ys[i]), int(xs[i])
    px = (xx + 0.5) / w; py = (yy + 0.5) / h
    d = ray_dir(px, py)
    hit, loc, nrm, idx, obj, mtx = scene.ray_cast(dg, origin, d)
    if not hit or obj is None:
        mat_over["<miss/world>"] += float(over[yy, xx]); mat_cnt["<miss/world>"] += 1
        continue
    mname = "<none>"
    try:
        if obj.material_slots and idx < len(obj.data.polygons):
            mi = obj.data.polygons[idx].material_index
            slot = obj.material_slots[mi] if mi < len(obj.material_slots) else None
            if slot and slot.material:
                mname = slot.material.name
                if mname not in mat_emissive:
                    m = slot.material
                    em = 0.0
                    if m.use_nodes:
                        for n in m.node_tree.nodes:
                            if n.bl_idname == "ShaderNodeEmission":
                                em = float(n.inputs["Strength"].default_value)
                            if n.bl_idname == "ShaderNodeBsdfPrincipled":
                                ei = n.inputs.get("Emission Strength")
                                if ei: em = max(em, float(ei.default_value))
                    mat_emissive[mname] = em
    except Exception:
        pass
    mat_over[mname] += float(over[yy, xx]); mat_cnt[mname] += 1

print("\nmaterial                          over_sum   n   emission_strength")
for mname, ov in sorted(mat_over.items(), key=lambda kv: -kv[1])[:20]:
    em = mat_emissive.get(mname, "-")
    print(f"  {mname:30s} {ov:8.3f}  {mat_cnt[mname]:3d}   {em}")
print("=== DONE ===", flush=True)
