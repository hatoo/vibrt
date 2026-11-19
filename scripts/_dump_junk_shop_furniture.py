"""Dump junk_shop's Furniture material and the Background Blue area light
for diagnosis of the 1000x over-delivery in vibrt."""
import bpy
from mathutils import Vector

scene = bpy.context.scene

# Material
m = bpy.data.materials.get("Furniture")
print(f"=== material 'Furniture' ===")
nt = m.node_tree
out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output), None)


def fmt(v):
    if hasattr(v, "__iter__"):
        return tuple(round(float(x), 4) for x in v)
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    return v


def walk(node, depth=0, seen=None):
    if seen is None:
        seen = set()
    if node.name in seen:
        print(f"{'  ' * depth}↻ {node.bl_idname}/{node.name}")
        return
    seen.add(node.name)
    label = f"{node.bl_idname}/{node.name}"
    if hasattr(node, "image") and node.image:
        label += f" image='{node.image.name}' cs={node.image.colorspace_settings.name}"
    if hasattr(node, "operation"):
        label += f" op={node.operation}"
    if hasattr(node, "blend_type"):
        label += f" blend={node.blend_type}"
    print(f"{'  ' * depth}{label}")
    for inp in node.inputs:
        if inp.is_linked:
            src = inp.links[0].from_node
            ssock = inp.links[0].from_socket
            print(f"{'  ' * (depth + 1)}in '{inp.name}' ← {src.bl_idname}/{src.name}.{ssock.name}")
            walk(src, depth + 2, seen)
        else:
            try:
                v = inp.default_value
                if hasattr(v, '__iter__') or isinstance(v, (int, float)):
                    print(f"{'  ' * (depth + 1)}in '{inp.name}' = {fmt(v)}")
            except Exception:
                pass


surf = out.inputs.get("Surface")
if surf and surf.is_linked:
    walk(surf.links[0].from_node)
vol = out.inputs.get("Volume")
if vol and vol.is_linked:
    print("\n  --- volume ---")
    walk(vol.links[0].from_node)

# Background Blue light
print(f"\n\n=== light 'Background Blue' ===")
lobj = bpy.data.objects.get("Background Blue")
if lobj:
    L = lobj.data
    print(f"  type = {L.type}")
    print(f"  energy = {L.energy}")
    print(f"  color = {tuple(round(float(x), 3) for x in L.color)}")
    if L.type == "AREA":
        print(f"  shape = {L.shape}")
        print(f"  size_x = {L.size}")
        if hasattr(L, 'size_y'):
            print(f"  size_y = {L.size_y}")
    print(f"  loc = {tuple(round(float(x), 2) for x in lobj.location)}")
    print(f"  rotation = {tuple(round(float(x), 2) for x in lobj.rotation_euler)}")
    print(f"  visible_camera = {lobj.visible_camera}")
    print(f"  visible_diffuse = {lobj.visible_diffuse}")

# Camera position vs Furniture vs Background Blue
print(f"\n\n=== distances ===")
cam = scene.camera
print(f"  camera loc = {tuple(round(float(x), 2) for x in cam.matrix_world.translation)}")
furn = bpy.data.objects.get("Furniture")
if furn:
    print(f"  Furniture loc = {tuple(round(float(x), 2) for x in furn.location)}")
    print(f"  Furniture dim = {tuple(round(float(x), 2) for x in furn.dimensions)}")
    print(f"  Furniture bbox center = {tuple(round(float(x), 2) for x in furn.matrix_world.translation)}")
if lobj:
    cam_to_light = (lobj.matrix_world.translation - cam.matrix_world.translation).length
    if furn:
        furn_to_light = (lobj.matrix_world.translation - furn.matrix_world.translation).length
        print(f"  Furniture → Background Blue distance = {furn_to_light:.2f}m")
    print(f"  Camera → Background Blue distance = {cam_to_light:.2f}m")
