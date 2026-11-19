"""Verify the prebaked-chain fix: export each fallback material's Base Color
texture via the real export_image_texture path and report the linear mean of
the resulting texture. Expected after fix: brick means dimmed by the terminal
HueSaturation Value (~0.25x), not the raw prebake mean.

Run: blender -b lone_monk.blend --python scripts/_diag_verify_prebake_chain.py
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
from vibrt_blender.bin_writer import BinWriter  # noqa: E402

TARGETS = ["brick marble smooth", "brick clay", "brick marble", "column marble",
           "leather - book - cover 1"]


def basecolor_sock(mat):
    for n in mat.node_tree.nodes:
        if n.bl_idname == "ShaderNodeBsdfPrincipled":
            return n.inputs.get("Base Color")
    return None


def tex_mean_from_writer(writer, desc):
    # In-process writer stores pixels in an array list; pull by array_index.
    ai = desc.get("array_index")
    if ai is None:
        return None
    arr = writer.texture_arrays[ai]
    a = np.asarray(arr, dtype=np.float32)
    w, h = desc["width"], desc["height"]
    return a.reshape(h, w, 4)[..., :3].reshape(-1, 3).mean(axis=0)


for nm in TARGETS:
    mat = bpy.data.materials.get(nm)
    if mat is None:
        print(f"### {nm!r}: NOT FOUND")
        continue
    bc = basecolor_sock(mat)
    writer = BinWriter()
    textures = []
    img, chain = ME._socket_linked_image_with_chain(bc)
    if img is None:
        print(f"### {nm!r}: no fallback image")
        continue
    if isinstance(img, ME._PreBakedTexture):
        prebake_mean = np.asarray(img.rgb, dtype=np.float32).reshape(-1, 3).mean(axis=0)
    else:
        prebake_mean = None
    idx = ME.export_image_texture(img, writer, textures, "srgb", chain=chain)
    desc = textures[idx]
    out_mean = tex_mean_from_writer(writer, desc)
    chain_ids = [c[0][0] if isinstance(c[0], tuple) else c[0] for c in chain]
    print(f"### {nm!r}  chain={chain_ids}")
    print(f"    prebake mean (raw)     : {prebake_mean}")
    print(f"    EXPORTED tex mean      : {out_mean}")
    if prebake_mean is not None and out_mean is not None:
        print(f"    dim factor (out/raw)   : {out_mean / np.maximum(prebake_mean, 1e-9)}")

print("=== DONE ===", flush=True)
