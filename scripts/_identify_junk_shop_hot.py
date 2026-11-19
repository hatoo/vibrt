"""Identify the object/material at junk_shop's hot pixel cluster
(centered around y=109, x=169 in a 500x250 image)."""
import math
import bpy
from mathutils import Vector

scene = bpy.context.scene
W, H = 500, 250
cam_obj = scene.camera
cam = cam_obj.data


def pixel_to_dir(py, px):
    nx = (px + 0.5) / W
    ny = (py + 0.5) / H
    if cam.sensor_fit == "VERTICAL":
        ty = math.tan(cam.angle / 2); tx = ty * (W / H)
    else:
        tx = math.tan(cam.angle / 2); ty = tx * (H / W)
    cx = (2 * nx - 1) * tx
    cy = (2 * ny - 1) * ty
    local = Vector((cx, cy, -1)).normalized()
    return cam_obj.matrix_world.translation, (cam_obj.matrix_world.to_3x3() @ local).normalized()


depsgraph = bpy.context.evaluated_depsgraph_get()

for label, py, px in [
    ("hot_center", 109, 169),
    ("hot_top", 106, 169),
    ("hot_bot", 112, 169),
    ("hot_left", 109, 165),
    ("hot_right", 109, 175),
    ("near", 100, 169),
    ("far", 130, 169),
]:
    o, d = pixel_to_dir(py, px)
    succ, loc, n, idx, obj, mw = scene.ray_cast(depsgraph, o, d)
    if succ:
        mat_name = "<none>"
        if obj.type == "MESH":
            try:
                ev = obj.evaluated_get(depsgraph)
                p = ev.data.polygons[idx] if idx < len(ev.data.polygons) else None
                if p is not None:
                    mi = p.material_index
                    if mi < len(obj.material_slots):
                        ms = obj.material_slots[mi]
                        if ms.material:
                            mat_name = ms.material.name
            except Exception as e:
                mat_name = f"<err {e}>"
        print(f"  {label:>10} ({py:3},{px:3}): hit '{obj.name}' @ {(loc - o).length:.2f}m  mat='{mat_name}'  normal=({n.x:.2f},{n.y:.2f},{n.z:.2f})")
    else:
        print(f"  {label:>10} ({py:3},{px:3}): no hit (escape)")

# Also check what materials are emissive in junk_shop
print("\n=== Emissive materials in junk_shop ===")
for mat in bpy.data.materials:
    if mat.node_tree is None:
        continue
    for n in mat.node_tree.nodes:
        if n.bl_idname == "ShaderNodeEmission":
            col = n.inputs.get("Color"); s = n.inputs.get("Strength")
            col_v = "linked" if col.is_linked else tuple(round(float(x), 3) for x in col.default_value[:3])
            s_v = "linked" if s.is_linked else round(float(s.default_value), 3)
            if (s_v == "linked" or s_v > 0) and (col_v == "linked" or any(x > 0 for x in col_v)):
                print(f"  '{mat.name}': Emission color={col_v} strength={s_v}")
                break
        elif n.bl_idname == "ShaderNodeBsdfPrincipled":
            col = n.inputs.get("Emission Color"); s = n.inputs.get("Emission Strength")
            if col is None or s is None:
                continue
            col_v = "linked" if col.is_linked else tuple(round(float(x), 3) for x in col.default_value[:3])
            s_v = "linked" if s.is_linked else round(float(s.default_value), 3)
            if (s_v == "linked" or s_v > 0) and (col_v == "linked" or any(x > 0 for x in col_v)):
                print(f"  '{mat.name}': PrincipledEmission color={col_v} strength={s_v}")
                break

# Lights and volume
print(f"\n=== Lights ===")
for obj in scene.objects:
    if obj.type == "LIGHT":
        print(f"  {obj.name}: type={obj.data.type}, energy={obj.data.energy}, color={tuple(round(float(x), 2) for x in obj.data.color)}")

print(f"\n=== World ===")
w = scene.world
print(f"  use_nodes={w.use_nodes}")
if w.node_tree:
    for n in w.node_tree.nodes:
        print(f"    {n.bl_idname}/{n.name}")

print(f"\n=== Cycles settings ===")
cy = scene.cycles
for attr in ("samples", "use_adaptive_sampling", "diffuse_bounces", "glossy_bounces",
             "transmission_bounces", "volume_bounces", "max_bounces", "blur_glossy",
             "sample_clamp_direct", "sample_clamp_indirect", "use_light_tree"):
    if hasattr(cy, attr):
        print(f"  {attr} = {getattr(cy, attr)}")
