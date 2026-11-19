"""Find every evaluated mesh whose bounding box CONTAINS the camera origin.
A surround mesh + clip_start=2 is the classic Blender 'tone-the-view-frustum'
trick that flat-archiviz might be using."""
import bpy
from mathutils import Vector

scene = bpy.context.scene
deps = bpy.context.evaluated_depsgraph_get()
eye = scene.camera.matrix_world.translation
print(f"camera eye: {tuple(round(x,3) for x in eye)}")

found = []
for obj in scene.objects:
    if obj.type != 'MESH':
        continue
    bb_local = [Vector(c) for c in obj.bound_box]
    bb_world = [obj.matrix_world @ c for c in bb_local]
    xs = [v.x for v in bb_world]; ys = [v.y for v in bb_world]; zs = [v.z for v in bb_world]
    if (min(xs) <= eye.x <= max(xs)
            and min(ys) <= eye.y <= max(ys)
            and min(zs) <= eye.z <= max(zs)):
        size_x = max(xs) - min(xs)
        size_y = max(ys) - min(ys)
        size_z = max(zs) - min(zs)
        n_verts = len(obj.data.vertices)
        mats = [s.material.name if s.material else "?" for s in obj.material_slots]
        # Also check if any material slot is emissive.
        emissive_flags = []
        for s in obj.material_slots:
            if s.material and s.material.use_nodes:
                emit = False
                for n in s.material.node_tree.nodes:
                    if n.bl_idname == 'ShaderNodeEmission':
                        emit = True; break
                emissive_flags.append("E" if emit else "-")
            else:
                emissive_flags.append("-")
        vc = getattr(obj, 'visible_camera', True)
        found.append((max(size_x, size_y, size_z), obj.name, n_verts, mats, emissive_flags, vc))

found.sort()
print(f"\n=== meshes whose bbox contains the camera ({len(found)}) ===")
for sz, name, nv, mats, emit, vc in found[:25]:
    print(f"  size={sz:6.2f}  vc={int(vc)}  obj={name:40s}  verts={nv:7d}  mats={mats}  emit={emit}")
