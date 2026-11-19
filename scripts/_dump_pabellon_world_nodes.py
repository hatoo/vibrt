"""Dump pabellon's world node tree structure + the mean linear colour of
each Environment texture, so we can reason about the B-over/R-under shift
vs Cycles."""
import bpy, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
for _m in list(sys.modules):
    if _m == "vibrt_blender" or _m.startswith("vibrt_blender."):
        del sys.modules[_m]
from vibrt_blender import material_export as ME  # noqa

w = bpy.context.scene.world
nt = w.node_tree
print("=== world node tree ===")
for n in nt.nodes:
    line = f"[{n.bl_idname}] {n.name!r}"
    if n.bl_idname == "ShaderNodeMixRGB":
        line += f"  blend={n.blend_type} fac={n.inputs['Fac'].default_value:.4f}"
    if n.bl_idname == "ShaderNodeMixShader":
        line += f"  fac={n.inputs[0].default_value:.4f}"
    if n.bl_idname == "ShaderNodeBackground":
        line += f"  strength={n.inputs['Strength'].default_value:.4f}"
    if n.bl_idname == "ShaderNodeTexEnvironment":
        img = n.image
        line += f"  img={img.name if img else None} proj={getattr(n,'projection',None)} cs={img.colorspace_settings.name if img else '-'}"
    print(" ", line)
    for inp in n.inputs:
        if inp.is_linked:
            l = inp.links[0]
            print(f"      <- {inp.name} = {l.from_node.name!r}.{l.from_socket.name}")


def env_mean(node):
    img = node.image
    if img is None:
        return None
    ww, hh = img.size
    if ww == 0 or hh == 0:
        return None
    px = np.empty(ww * hh * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    rgb = px.reshape(hh, ww, 4)[..., :3].astype(np.float32)
    if img.colorspace_settings.name.lower().startswith("srgb"):
        rgb = ME._srgb_to_linear_np(rgb)
    return rgb.reshape(-1, 3).mean(axis=0)


print("\n=== env texture linear means ===")
for n in nt.nodes:
    if n.bl_idname == "ShaderNodeTexEnvironment":
        m = env_mean(n)
        print(f"  {n.name!r}: linear mean = {m}")

print("=== DONE ===", flush=True)
