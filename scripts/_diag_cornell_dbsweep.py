"""Build a minimal controlled scene (white floor + shadow-caster cube + area
light) and run a db-sweep in both engines to isolate the low-contrast
mechanism: is the dim (shadow/indirect) region's over-delivery from indirect
bounces or direct/ambient NEE?

Renders vibrt & cycles at diffuse_bounces=0 (direct only) and =4 (indirect),
compares the shadow region (dim) and lit region (bright).

Run: blender -b --factory-startup --python scripts/_diag_cornell_dbsweep.py -- <db>
"""
import sys, math
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
db = int(argv[argv.index("--") + 1]) if "--" in argv else 4

# --- build scene ---
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

def add_diffuse(name, color):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Metallic"].default_value = 0.0
    try: bsdf.inputs["Specular IOR Level"].default_value = 0.0
    except Exception: pass
    return m

# Floor
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
floor = bpy.context.active_object; floor.data.materials.append(add_diffuse("Floor", (0.7, 0.7, 0.7)))
# Shadow-caster cube (raised so it casts a shadow + creates indirect)
bpy.ops.mesh.primitive_cube_add(size=1.5, location=(0, 0, 0.75))
cube = bpy.context.active_object; cube.data.materials.append(add_diffuse("Cube", (0.7, 0.2, 0.2)))
# Area light above, offset so the cube casts a lateral shadow
bpy.ops.object.light_add(type='AREA', location=(2.5, 0, 5))
light = bpy.context.active_object; light.data.energy = 1000.0; light.data.size = 1.5
light.rotation_euler = (math.radians(25), 0, 0)
# Black world (isolate the area light; no ambient)
w = bpy.data.worlds.new("W"); w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
bpy.context.scene.world = w
# Camera looking down at the floor
bpy.ops.object.camera_add(location=(0, -6, 6), rotation=(math.radians(48), 0, 0))
cam = bpy.context.active_object; bpy.context.scene.camera = cam

s = bpy.context.scene
s.render.resolution_x = 400; s.render.resolution_y = 400
s.render.image_settings.file_format = "OPEN_EXR"; s.render.image_settings.color_depth = "32"
s.view_settings.view_transform = "Raw"; s.view_settings.look = "None"
s.view_settings.exposure = 0.0; s.view_settings.gamma = 1.0
s.render.use_compositing = False
OUT = HERE / "test_scenes" / "_eng_diff"
OUT.mkdir(parents=True, exist_ok=True)
SPP = 128


def render(engine):
    s.render.engine = "VIBRT" if engine == "vibrt" else "CYCLES"
    if engine == "vibrt":
        s.vibrt_spp = SPP
        try: s.vibrt_diffuse_bounces = db
        except Exception: pass
    else:
        s.cycles.samples = SPP; s.cycles.use_adaptive_sampling = False
        s.cycles.diffuse_bounces = db; s.cycles.max_bounces = max(db, 1)
        s.cycles.sample_clamp_direct = 0; s.cycles.sample_clamp_indirect = 0
    p = OUT / f"cornell_db{db}_{engine}.exr"
    s.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(p)); img.colorspace_settings.name = "Linear Rec.709"
    ww, hh = int(img.size[0]), int(img.size[1])
    fl = np.empty(ww*hh*4, np.float32); img.pixels.foreach_get(fl); bpy.data.images.remove(img)
    return fl.reshape(hh, ww, 4)[..., :3].reshape(-1, 3)


v = render("vibrt"); c = render("cycles")
clum = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
print(f"\n=== cornell db={db} SPP={SPP} ===")
print(f"overall ratio = {np.round(v.mean(0)/np.maximum(c.mean(0),1e-9),3)}")
for lo, hi, t in [(0.0001, 0.05, 'dim/shadow'), (0.05, 0.3, 'mid'), (0.3, 1e9, 'bright/lit')]:
    m = (clum >= lo) & (clum < hi)
    if m.sum() == 0: continue
    print(f"  {t:12s} n={int(m.sum()):6d} vibrt={np.round(v[m].mean(0),4)} cyc={np.round(c[m].mean(0),4)} ratio={np.round(v[m].mean(0)/np.maximum(c[m].mean(0),1e-9),3)}")
print("=== DONE ===", flush=True)
