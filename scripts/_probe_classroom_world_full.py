"""Walk the entire World node tree to make sure we're not missing
anything. Also dump the dayLight_portal mesh face normal direction."""
import bpy

scene = bpy.context.scene
world = scene.world
print(f"=== world {world.name!r} ===")
print(f"  use_nodes = {world.use_nodes}")
if world.node_tree:
    print(f"  --- all nodes ---")
    for n in world.node_tree.nodes:
        print(f"  {n.bl_idname}/{n.name}")
        for inp in n.inputs:
            try:
                v = inp.default_value
                if hasattr(v, "__iter__"):
                    v = tuple(round(float(x), 3) for x in v)
                else:
                    v = round(float(v), 3) if isinstance(v, (int, float)) else v
                src = inp.links[0].from_node if inp.is_linked else None
                if src:
                    print(f"    in {inp.name}: linked from {src.bl_idname}/{src.name}")
                else:
                    print(f"    in {inp.name} = {v}")
            except Exception:
                pass

# Now: classroom's `windows` mesh -> first face normal direction
# (in world space). Compare with sun direction and camera direction.
print(f"\n=== windows mesh face normal ===")
import mathutils
for obj_name in ("windows", "hallWindow"):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != "MESH":
        print(f"  {obj_name}: not found / not mesh")
        continue
    mesh = obj.data
    if not mesh.polygons:
        print(f"  {obj_name}: no polygons")
        continue
    # World transform
    m = obj.matrix_world.to_3x3()
    locs = []
    norms = []
    for p in mesh.polygons[:5]:
        # face normal in world space
        n_local = mathutils.Vector(p.normal)
        n_world = (m @ n_local).normalized()
        c_world = obj.matrix_world @ p.center
        locs.append(tuple(round(float(c), 3) for c in c_world))
        norms.append(tuple(round(float(c), 3) for c in n_world))
    print(f"  {obj_name}: poly_count={len(mesh.polygons)}")
    for loc, nor in zip(locs, norms):
        print(f"    poly@{loc} normal={nor}")
