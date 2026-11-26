"""Export pabellon and measure the two world layers (a=sky bake, b=flat image)
as vibrt ships them, then compute the vibrt mix mean and compare to Cycles'
rendered sky mean (0.317, 0.333, 0.320)."""
import bpy, os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
for _m in list(sys.modules):
    if _m == "vibrt_blender" or _m.startswith("vibrt_blender."):
        del sys.modules[_m]
import vibrt_blender  # noqa
from vibrt_blender.export_context import ExportContext
export_context = ExportContext()
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass
from vibrt_blender.exporter import export_scene_to_memory, prebake_sky_envmaps_for_world

prebake_sky_envmaps_for_world(bpy.context.scene.world, w=2048, h=1024, context=export_context)
deps = bpy.context.evaluated_depsgraph_get()
json_str, _bin, texs = export_scene_to_memory(deps, texture_pct=100, context=export_context)
data = json.loads(json_str)
world = data["world"]
print("world type:", world["type"], "fac:", world.get("fac"))
a = world["a"]; b = world["b"]
print("a:", {k: a[k] for k in a if k != "rotation"})
print("b:", {k: b[k] for k in b if k != "rotation"})

tdescs = data["textures"]


def tex_mean(idx):
    d = tdescs[idx]
    ai = d.get("array_index")
    arr = np.asarray(texs[ai], dtype=np.float32)
    w, h, ch = d["width"], d["height"], d["channels"]
    m = arr.reshape(h, w, ch)[..., :3].reshape(-1, 3).mean(axis=0)
    return m, d.get("colorspace"), (w, h)


ma, csa, sza = tex_mean(a["texture"])
mb, csb, szb = tex_mean(b["texture"])
print(f"\nlayer a (sky) tex#{a['texture']} cs={csa} size={sza} mean={np.round(ma,4)} strength={a['strength']}")
print(f"layer b (img) tex#{b['texture']} cs={csb} size={szb} mean={np.round(mb,4)} strength={b['strength']}")

fac = world["fac"]
# NOTE: naive mean-of-mix ignores per-direction FLAT projection remapping of b,
# but is a first-order check of the colour balance.
mix_mean = ma * a["strength"] * (1 - fac) + mb * b["strength"] * fac
print(f"\nvibrt mix mean (naive) = {np.round(mix_mean,4)}")
print(f"cycles rendered sky    = [0.317  0.3325 0.32  ]")
print(f"vibrt  rendered sky    = [0.2694 0.3552 0.5039]")

# Raw sRGB->linear mean of the source JPG for reference.
img = None
for n in bpy.context.scene.world.node_tree.nodes:
    if n.bl_idname == "ShaderNodeTexImage" and n.image:
        img = n.image
if img is not None:
    from vibrt_blender import material_export as ME
    w2, h2 = img.size
    px = np.empty(w2 * h2 * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    rgb = px.reshape(h2, w2, 4)[..., :3].astype(np.float32)
    raw_mean = rgb.reshape(-1, 3).mean(axis=0)
    lin_mean = ME._srgb_to_linear_np(rgb).reshape(-1, 3).mean(axis=0)
    print(f"\nsource JPG {img.name} cs={img.colorspace_settings.name}")
    print(f"  raw(sRGB-encoded) mean = {np.round(raw_mean,4)}")
    print(f"  sRGB->linear mean      = {np.round(lin_mean,4)}")
print("=== DONE ===", flush=True)
