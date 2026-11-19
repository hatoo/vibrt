"""Inspect the suspicious 'Plane' mesh (the 150-vertex terrain plane
that covers the whole pool area). Compare its raw mesh data with the
depsgraph-evaluated mesh, dump every modifier, and tabulate vertex Z
distribution so we can spot a missing displacement / subsurf step that
Cycles applies but vibrt doesn't."""
import bpy

scene = bpy.context.scene
obj = bpy.data.objects.get("Plane")
if obj is None:
    raise SystemExit("no object named 'Plane'")
me = obj.data

print(f"=== raw 'Plane' mesh ===")
print(f"  verts={len(me.vertices)} polys={len(me.polygons)} edges={len(me.edges)}")
print(f"  modifiers: {[m.type for m in obj.modifiers]}")
for m in obj.modifiers:
    print(f"  mod {m.type} '{m.name}' show_render={m.show_render}")
    for prop in m.bl_rna.properties:
        if prop.identifier in ("rna_type", "name", "type", "show_viewport",
                               "show_render", "show_in_editmode", "show_on_cage",
                               "show_expanded", "use_apply_on_spline"):
            continue
        try:
            v = getattr(m, prop.identifier)
            if v is None or isinstance(v, (bool, int, float, str)):
                print(f"    {prop.identifier} = {v}")
            elif hasattr(v, "name"):
                print(f"    {prop.identifier} = <{type(v).__name__} '{v.name}'>")
        except Exception:
            pass

# Per-vertex Z distribution (local + world).
local_zs = sorted(v.co.z for v in me.vertices)
mw = obj.matrix_world
world_zs = sorted((mw @ v.co).z for v in me.vertices)
print(f"\n=== raw mesh vertex Z ===")
print(f"  local: min={local_zs[0]:.4f} max={local_zs[-1]:.4f} median={local_zs[len(local_zs)//2]:.4f}")
print(f"  world: min={world_zs[0]:.4f} max={world_zs[-1]:.4f} median={world_zs[len(world_zs)//2]:.4f}")
# Histogram local Z
import collections
buckets = collections.Counter()
for z in local_zs:
    buckets[round(z, 2)] += 1
print(f"  local Z histogram (rounded 0.01): {dict(sorted(buckets.items())[:30])}")

# Now evaluate via depsgraph (so modifiers, edge subsurfacing, etc. apply).
deps = bpy.context.evaluated_depsgraph_get()
oe = obj.evaluated_get(deps)
ev = oe.to_mesh(depsgraph=deps)
try:
    print(f"\n=== depsgraph-evaluated 'Plane' mesh ===")
    print(f"  verts={len(ev.vertices)} polys={len(ev.polygons)}")
    if ev.vertices:
        ezs = sorted(v.co.z for v in ev.vertices)
        wzs = sorted((mw @ v.co).z for v in ev.vertices)
        print(f"  local Z: min={ezs[0]:.4f} max={ezs[-1]:.4f} median={ezs[len(ezs)//2]:.4f}")
        print(f"  world Z: min={wzs[0]:.4f} max={wzs[-1]:.4f} median={wzs[len(wzs)//2]:.4f}")
finally:
    oe.to_mesh_clear()

# Dump material list.
print(f"\n=== materials on 'Plane' ===")
for s in obj.material_slots:
    print(f"  {s.material.name if s.material else None}")
