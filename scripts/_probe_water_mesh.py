"""Inspect water plane geometry: face count, normal direction, etc."""
import bpy

obj = bpy.data.objects.get("water_plane_still")
print(f"object: {obj.name}")
print(f"  location: {tuple(round(x,3) for x in obj.location)}")
print(f"  rotation: {tuple(round(x,3) for x in obj.rotation_euler)}")
print(f"  scale: {tuple(round(x,3) for x in obj.scale)}")
print(f"  display_type: {obj.display_type}")
me = obj.data
print(f"  vertices: {len(me.vertices)}")
print(f"  edges: {len(me.edges)}")
print(f"  polygons: {len(me.polygons)}")
print(f"  materials: {[s.material.name if s.material else None for s in obj.material_slots]}")
print(f"  hide_render: {obj.hide_render}")
print(f"  visible_camera: {getattr(obj, 'visible_camera', 'n/a')}")

# Polygon normals (in object space)
for i, p in enumerate(me.polygons):
    if i >= 5: break
    print(f"  poly[{i}]: normal={tuple(round(x,3) for x in p.normal)} verts={list(p.vertices)}")

# Vertex z values (in object space, before world transform)
zs = sorted({round(v.co.z, 4) for v in me.vertices})
print(f"  vertex z values (object-local): {zs[:5]}")

# World-space vertex z after applying obj.matrix_world
wz = sorted({round((obj.matrix_world @ v.co).z, 4) for v in me.vertices})
print(f"  vertex z values (world): {wz[:5]}")
