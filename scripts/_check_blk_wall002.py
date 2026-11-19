"""Check blk_wall.002's visibility and material settings — both hot pixels'
ray_cast hit this same object, but it might be camera-invisible."""
import bpy

scene = bpy.context.scene
for name in ("blk_wall.002", "blk_wall.001", "blk_wall.003"):
    obj = bpy.data.objects.get(name)
    if obj is None:
        print(f"missing: {name}")
        continue
    print(f"\n=== {name} ===")
    print(f"  hide_render        = {obj.hide_render}")
    print(f"  visible_camera     = {obj.visible_camera}")
    print(f"  visible_diffuse    = {obj.visible_diffuse}")
    print(f"  visible_glossy     = {obj.visible_glossy}")
    print(f"  visible_shadow     = {obj.visible_shadow}")
    print(f"  visible_transmission= {obj.visible_transmission}")
    print(f"  is_holdout         = {getattr(obj, 'is_holdout', '?')}")
    print(f"  cycles.is_shadow_catcher = {getattr(obj.cycles, 'is_shadow_catcher', '?')}")
    print(f"  poly count         = {len(obj.data.polygons)}")
    print(f"  bbox center        = {tuple(round(x, 2) for x in obj.location)}")
    print(f"  dimensions         = {tuple(round(x, 2) for x in obj.dimensions)}")
    print(f"  materials:")
    for i, ms in enumerate(obj.material_slots):
        if ms.material:
            print(f"    [{i}] {ms.material.name}")
        else:
            print(f"    [{i}] <none>")

# Now do a ray_cast with cycles-style visible_camera filter, by hiding
# blk_wall.002 first then ray_cast again
print("\n\n=== Ray cast with blk_wall.002 hidden ===")
target = bpy.data.objects.get("blk_wall.002")
if target:
    target.hide_set(True)
    target.hide_viewport = True
    # Force depsgraph update
    bpy.context.view_layer.update()

import math
from mathutils import Vector
W, H = 360, 270
cam_obj = scene.camera
cam = cam_obj.data

def pixel_to_dir(py, px):
    nx = (px + 0.5) / W; ny = (py + 0.5) / H
    if cam.sensor_fit == "VERTICAL":
        ty = math.tan(cam.angle / 2); tx = ty * (W / H)
    else:
        tx = math.tan(cam.angle / 2); ty = tx * (H / W)
    cx = (2 * nx - 1) * tx; cy = (2 * ny - 1) * ty
    local = Vector((cx, cy, -1)).normalized()
    return cam_obj.matrix_world.translation, (cam_obj.matrix_world.to_3x3() @ local).normalized()


depsgraph = bpy.context.evaluated_depsgraph_get()
for label, py, px in [("hot1", 166, 200), ("hot2", 185, 142)]:
    o, d = pixel_to_dir(py, px)
    succ, loc, n, idx, obj, mat_world = scene.ray_cast(depsgraph, o, d)
    print(f"\n  {label} ({py},{px}): direction=({d.x:.3f},{d.y:.3f},{d.z:.3f})")
    if succ:
        print(f"    → hit '{obj.name}' at d={(loc - o).length:.2f}m, normal=({n.x:.2f},{n.y:.2f},{n.z:.2f})")
        if obj.type == "MESH":
            # find material at hit polygon
            ev = obj.evaluated_get(depsgraph)
            try:
                p = ev.data.polygons[idx]
                mi = p.material_index
                ms = obj.material_slots[mi] if mi < len(obj.material_slots) else None
                print(f"    material = {ms.material.name if ms and ms.material else '<none>'}")
            except Exception as e:
                print(f"    material lookup err: {e}")
    else:
        print(f"    → no hit (escapes sky)")
