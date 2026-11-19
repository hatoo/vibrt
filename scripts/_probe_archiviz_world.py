"""Probe the flat-archiviz scene: world setup + objects close to the camera +
their visibility flags."""
import bpy

scene = bpy.context.scene
cam = scene.camera
eye = cam.matrix_world.translation

print("=== world ===")
w = scene.world
print(f"world.use_nodes = {w.use_nodes}")
if w.use_nodes:
    for n in w.node_tree.nodes:
        info = f"  {n.bl_idname:35s} '{n.name}'"
        if hasattr(n, 'image') and getattr(n, 'image', None):
            info += f"  image='{n.image.name}'"
        # default values
        for s in n.inputs:
            if not s.is_linked and hasattr(s, 'default_value'):
                v = s.default_value
                try:
                    v = list(v) if hasattr(v, '__len__') else v
                except Exception:
                    pass
                info += f"  [{s.name}={v}]"
        print(info)

print()
print(f"=== camera at world {tuple(round(x,3) for x in eye)} ===")
print(f"  visible_camera = {getattr(cam, 'visible_camera', 'N/A')}")

# Find geometry within 2.5m of the camera (closer than clip_start +ε).
print()
print("=== mesh objects within 2.5m of camera (sorted by distance) ===")
near = []
for obj in scene.objects:
    if obj.type != 'MESH':
        continue
    o = obj.matrix_world.translation
    d = (o - eye).length
    if d < 2.5:
        near.append((d, obj))
near.sort()
for (d, obj) in near[:30]:
    vc = getattr(obj, 'visible_camera', True)
    vg = getattr(obj, 'visible_glossy', True)
    print(f"  {d:5.2f}m  vc={int(vc)} vg={int(vg)}  {obj.name}  ({len(obj.data.vertices)} verts)")
print(f"  total within 2.5m: {len(near)}")
