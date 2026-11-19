"""Isolate bmw27's under-delivery: render both engines with emissive
materials forced to PURE emission (strip glass/transmission/metallic). If the
0.38x under in bright regions closes, the glass+emission delivery is the bug.

Run: blender -b bmw27_gpu.blend --python scripts/_diag_bmw27_glasstest.py -- <baseline|pure>
"""
import sys
from pathlib import Path
import numpy as np
import bpy

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

argv = sys.argv
mode = argv[argv.index("--") + 1] if "--" in argv else "baseline"

EMISSIVE = {"Emit", "HeadLightAngelEye", "HeadLightGlass"}
if mode == "pure":
    for nm in EMISSIVE:
        m = bpy.data.materials.get(nm)
        if not m or not m.use_nodes:
            continue
        nt = m.node_tree
        # Find the emission color/strength currently feeding output, rebuild
        # as a single Emission shader (no glass).
        em = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeEmission"), None)
        out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
        if em and out:
            nt.links.new(em.outputs[0], out.inputs["Surface"])
            print(f"  {nm}: forced pure Emission (str={em.inputs['Strength'].default_value})")

OUT = HERE / "test_scenes" / "_eng_diff"
SPP, PCT = 64, 25


def cfg(engine):
    s = bpy.context.scene
    s.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        s.vibrt_spp = SPP
    else:
        s.cycles.samples = SPP; s.cycles.use_adaptive_sampling = False
    s.render.resolution_percentage = PCT
    s.render.image_settings.file_format = "OPEN_EXR"; s.render.image_settings.color_depth = "32"
    s.view_settings.view_transform = "Raw"; s.view_settings.look = "None"
    s.view_settings.exposure = 0.0; s.view_settings.gamma = 1.0
    s.render.use_compositing = False


def render(engine):
    cfg(engine)
    p = OUT / f"bmw_gt_{mode}_{engine}.exr"
    bpy.context.scene.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(p)); img.colorspace_settings.name = "Linear Rec.709"
    w, h = int(img.size[0]), int(img.size[1])
    fl = np.empty(w*h*4, np.float32); img.pixels.foreach_get(fl); bpy.data.images.remove(img)
    return fl.reshape(h, w, 4)[..., :3].reshape(-1, 3)


v = render("vibrt"); c = render("cycles")
vlum = v @ np.array([0.2126,0.7152,0.0722],np.float32)
clum = c @ np.array([0.2126,0.7152,0.0722],np.float32)
print(f"\nmode={mode} overall ratio={np.round(v.mean(0)/np.maximum(c.mean(0),1e-9),3)}")
for lo,hi in [(0,0.5),(0.5,3),(3,1e9)]:
    m=(clum>=lo)&(clum<hi);
    if m.sum()==0: continue
    print(f"  bucket[{lo},{hi}) n={int(m.sum())} ratio={np.round(v[m].mean(0)/np.maximum(c[m].mean(0),1e-9),3)} vibrt={np.round(v[m].mean(0),2)} cyc={np.round(c[m].mean(0),2)}")
print("=== DONE ===", flush=True)
