"""For lone_monk 'fallback' materials, show the baked transform chain that
vibrt's `_socket_linked_image_with_chain` produces, and measure the effect
on the texture mean. Compare against the *full* Cycles chain mean (terminal
HueSaturation Value applied) computed by evaluating the node tree faithfully.

Run: blender -b lone_monk.blend --python scripts/_diag_fallback_bake.py -- "brick marble smooth" ...
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "blender"))
for _m in list(sys.modules):
    if _m == "vibrt_blender" or _m.startswith("vibrt_blender."):
        del sys.modules[_m]

import bpy  # noqa: E402
from vibrt_blender import material_export as ME  # noqa: E402

argv = sys.argv
targets = argv[argv.index("--") + 1:] if "--" in argv else \
    ["brick marble smooth", "brick clay", "brick marble", "column marble"]


def basecolor_sock(mat):
    for n in mat.node_tree.nodes:
        if n.bl_idname == "ShaderNodeBsdfPrincipled":
            return n.inputs.get("Base Color")
    return None


def tex_linear_mean(img):
    w, h = img.size
    if w == 0 or h == 0:
        return None
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    rgb = px.reshape(h, w, 4)[..., :3].astype(np.float32)
    srgb = img.colorspace_settings.name.lower().startswith("srgb")
    if srgb:
        rgb = ME._srgb_to_linear_np(rgb)
    return rgb.reshape(-1, 3).mean(axis=0)


for nm in targets:
    mat = bpy.data.materials.get(nm)
    if mat is None:
        print(f"\n### {nm!r}: NOT FOUND")
        continue
    bc = basecolor_sock(mat)
    print(f"\n### {nm!r}")
    img, chain = ME._socket_linked_image_with_chain(bc)
    if img is None:
        print("  fallback found NO image")
        continue
    if isinstance(img, ME._PreBakedTexture):
        print(f"  fallback PREBAKED tex {img.w}x{img.h} (already linear, chain ignored on bake)")
        base_mean = np.asarray(img.rgb, dtype=np.float32).reshape(-1, 3).mean(axis=0)
    else:
        print(f"  fallback image: {img.name!r}  size={tuple(img.size)} cs={img.colorspace_settings.name}")
        base_mean = tex_linear_mean(img)
    print(f"  chain ({len(chain)} transforms):")
    for cid, fn in chain:
        print(f"    - {cid}")
    print(f"  texture linear mean      : {base_mean}")
    # Apply vibrt's baked chain to the texture mean (chain ops are linear-space).
    v = base_mean.reshape(1, 1, 3).astype(np.float32)
    for cid, fn in chain:
        v = fn(v)
    vibrt_out = v.reshape(3)
    print(f"  vibrt baked-chain mean   : {vibrt_out}")
    # Terminal HueSaturation on the Base Color socket = the intended dimming.
    src = ME._follow_reroutes(bc).links[0].from_node
    if src.bl_idname == "ShaderNodeHueSaturation":
        hue = src.inputs["Hue"].default_value
        sat = src.inputs["Saturation"].default_value
        val = src.inputs["Value"].default_value
        fac = src.inputs["Fac"].default_value
        from vibrt_blender import color_fold as CF
        correct = CF._eval_hue_sat(list(vibrt_out), hue, sat, val, fac)
        print(f"  terminal HueSat(V={val:.3f}) applied to vibrt mean: {correct}")
        ratio = vibrt_out / np.maximum(np.array(correct), 1e-9)
        print(f"  >>> vibrt/correct ratio  : {ratio}  (if ~4x, Value is being DROPPED)")

print("\n=== DONE ===", flush=True)
