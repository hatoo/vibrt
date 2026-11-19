"""Localize bmw27's under-delivery. Bucket by cycles luminance, report where
vibrt is dim, and ray_cast the worst-under bright pixels to their materials.
Two modes: 'analyze' (EXR only, fast) and 'raycast' (needs blend loaded).
"""
import sys
from pathlib import Path
import numpy as np
import bpy

HERE = Path(__file__).resolve().parent.parent
VE = HERE / "test_scenes" / "_eng_diff" / "bmw27_vibrt.exr"
CE = HERE / "test_scenes" / "_eng_diff" / "bmw27_cycles.exr"
argv = sys.argv
mode = argv[argv.index("--") + 1] if "--" in argv else "analyze"


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
vf = v.reshape(-1, 3); cf = c.reshape(-1, 3)
vlum = vf @ np.array([0.2126, 0.7152, 0.0722], np.float32)
clum = cf @ np.array([0.2126, 0.7152, 0.0722], np.float32)
print(f"bmw27 {w}x{h}  overall ratio={np.round(vf.mean(0)/np.maximum(cf.mean(0),1e-9),3)}")

print("\nbucket (cycles lum)   n       vibrt_mean   cycles_mean   ratio  under_energy_share")
edges = [0, 0.05, 0.2, 0.5, 1.0, 3.0, 1e9]
under_tot = 0.0
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (clum >= lo) & (clum < hi)
    n = int(m.sum())
    if n == 0:
        continue
    vm = vf[m].mean(0); cm = cf[m].mean(0)
    under = (cf[m].sum(0) - vf[m].sum(0)).sum()
    under_tot += max(under, 0)
    rows.append((lo, hi, n, vm, cm, under))
for lo, hi, n, vm, cm, under in rows:
    share = max(under, 0) / max(under_tot, 1e-9)
    print(f"[{lo:.2f},{hi:.2f})        {n:7d}  {np.round(vm,3)} {np.round(cm,3)} {vm.mean()/max(cm.mean(),1e-9):.2f}   {share:+.1%}")

if mode == "raycast":
    from collections import defaultdict
    under = clum - vlum
    mask2d = ((clum > 0.3) & (under > 0.2)).reshape(h, w)
    ys, xs = np.where(mask2d)
    o = (clum - vlum).reshape(h, w)
    order = np.argsort(-o[ys, xs]); ys, xs = ys[order], xs[order]
    scene = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    cam = scene.camera; cm = cam.matrix_world
    tr, br, bl, tl = [cm @ p for p in cam.data.view_frame(scene=scene)]
    origin = cm.translation
    def rd(px, py):
        top = tl.lerp(tr, px); bot = bl.lerp(br, px)
        return (bot.lerp(top, 1 - py) - origin).normalized()
    mat_u = defaultdict(float); mat_n = defaultdict(int); mat_info = {}
    N = min(400, len(ys))
    print(f"\nbright under-delivering pixels: {len(ys)} (top {N})")
    for i in range(N):
        yy, xx = int(ys[i]), int(xs[i])
        d = rd((xx+0.5)/w, (yy+0.5)/h)
        hit, loc, nrm, idx, obj, mtx = scene.ray_cast(dg, origin, d)
        nm = "<miss/world>"
        if hit and obj and obj.material_slots and idx < len(obj.data.polygons):
            mi = obj.data.polygons[idx].material_index
            slot = obj.material_slots[mi] if mi < len(obj.material_slots) else None
            if slot and slot.material:
                nm = slot.material.name
                if nm not in mat_info:
                    m = slot.material; info = {}
                    if m.use_nodes:
                        for nn in m.node_tree.nodes:
                            if nn.bl_idname == "ShaderNodeBsdfPrincipled":
                                for k in ("Metallic","Roughness","Transmission Weight","Coat Weight"):
                                    ii = nn.inputs.get(k)
                                    if ii and not ii.is_linked: info[k]=round(float(ii.default_value),3)
                            info["node"] = info.get("node","") + nn.bl_idname.replace("ShaderNode","")+" "
                    mat_info[nm] = info
        mat_u[nm] += float(o[yy, xx]); mat_n[nm] += 1
    print("material                     under_sum  n   info")
    for nm, uu in sorted(mat_u.items(), key=lambda kv:-kv[1])[:12]:
        print(f"  {nm:26s} {uu:8.3f} {mat_n[nm]:3d}  {mat_info.get(nm,{})}")
print("=== DONE ===", flush=True)
