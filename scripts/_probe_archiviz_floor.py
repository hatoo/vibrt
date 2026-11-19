"""Find the mesh/material at world (4.6, 9, 0) — the floor area where the
flat-archiviz renderer's left-bottom corner ray lands. Inspect the material's
shader graph (especially the Math.010 node that vibrt warned about)."""
import bpy
from mathutils import Vector

# Walk every mesh; find the one whose bounding box contains the target.
target = Vector((4.6, 9.0, 0.0))
print(f"target world position: {tuple(target)}")
candidates = []
for obj in bpy.context.scene.objects:
    if obj.type != 'MESH':
        continue
    bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [v.x for v in bb]; ys = [v.y for v in bb]; zs = [v.z for v in bb]
    if (min(xs)-0.1 <= target.x <= max(xs)+0.1
            and min(ys)-0.1 <= target.y <= max(ys)+0.1
            and min(zs)-0.1 <= target.z <= max(zs)+0.1):
        size = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
        candidates.append((size, obj))
candidates.sort()
for sz, obj in candidates[:8]:
    mats = [s.material.name if s.material else "(none)" for s in obj.material_slots]
    print(f"  size={sz:6.2f}  obj={obj.name:40s}  mats={mats}")

# Show the Chocofur_Concrete_Solid_07 material graph
mat = bpy.data.materials.get("Floor")
if mat is None:
    print("Chocofur_Concrete_Solid_07 not found")
else:
    print(f"\n=== material '{mat.name}' nodes ===")
    if not mat.use_nodes:
        print("  (no nodes)")
    else:
        nt = mat.node_tree
        for n in nt.nodes:
            print(f"  {n.bl_idname:35s} '{n.name}'", end="")
            if n.bl_idname == 'ShaderNodeMath':
                print(f"  op={n.operation}")
            elif hasattr(n, 'image') and getattr(n, 'image', None):
                print(f"  image='{n.image.name}'")
            else:
                print()
        print("\n=== links ===")
        for L in nt.links:
            print(f"  {L.from_node.name}.{L.from_socket.name}  ->  {L.to_node.name}.{L.to_socket.name}")
