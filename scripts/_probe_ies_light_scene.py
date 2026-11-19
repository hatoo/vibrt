"""Probe Blender's official Cycles IES regression scene
(`render/light/ies_light.blend` from blender-test-data) to understand
its layout: lights, IES nodes, mesh / materials, camera, render
resolution. Used to plan vibrt's IES support implementation.
"""
import bpy

print("=== Render settings ===")
r = bpy.context.scene.render
print(f"  resolution: {r.resolution_x}x{r.resolution_y} @ {r.resolution_percentage}%")
print(f"  engine: {bpy.context.scene.render.engine}")
cy = getattr(bpy.context.scene, "cycles", None)
if cy is not None:
    print(f"  cycles.samples: {cy.samples}")
    print(f"  cycles.diffuse_bounces: {cy.diffuse_bounces}, glossy: {cy.glossy_bounces}, "
          f"transmission: {cy.transmission_bounces}")

print("\n=== Camera ===")
cam = bpy.context.scene.camera
if cam is not None:
    print(f"  {cam.name}: pos={tuple(cam.location)} rot_euler={tuple(cam.rotation_euler)}")
    if cam.data.type == 'PERSP':
        print(f"  perspective, fov_x={cam.data.angle:.4f}rad")

print("\n=== Lights ===")
for obj in bpy.context.scene.objects:
    if obj.type != 'LIGHT':
        continue
    light = obj.data
    print(f"  [{obj.name}] type={light.type} color={tuple(light.color)} energy={light.energy}")
    print(f"    pos={tuple(obj.location)} rot_euler={tuple(obj.rotation_euler)}")
    print(f"    use_nodes={light.use_nodes}")
    if not light.use_nodes or light.node_tree is None:
        continue
    for n in light.node_tree.nodes:
        print(f"    node {n.name!r} bl_idname={n.bl_idname}")
        if n.bl_idname == 'ShaderNodeTexIES':
            print(f"      mode={n.mode}")
            print(f"      filepath={n.filepath!r}")
            if n.ies is not None:
                txt = n.ies.as_string()
                print(f"      ies (Text datablock): name={n.ies.name!r} size={len(txt)} chars")
                print(f"      first 200 chars:\n        {txt[:200].replace(chr(10), ' / ')}")
            else:
                print(f"      ies datablock: None")

print("\n=== Meshes ===")
for obj in bpy.context.scene.objects:
    if obj.type != 'MESH':
        continue
    me = obj.data
    print(f"  [{obj.name}] verts={len(me.vertices)} polys={len(me.polygons)} "
          f"materials={[m.name if m else None for m in me.materials]}")

print("\n=== bpy.data.texts (potential internal IES) ===")
for t in bpy.data.texts:
    s = t.as_string()
    print(f"  {t.name!r}: filepath={t.filepath!r} size={len(s)} chars")
    if "TILT=" in s:
        print(f"    looks like an IES file. First 200 chars: {s[:200].replace(chr(10), ' / ')}")

print("\n=== World ===")
w = bpy.context.scene.world
if w is not None:
    print(f"  {w.name}: use_nodes={w.use_nodes}")
    if w.use_nodes and w.node_tree is not None:
        for n in w.node_tree.nodes:
            print(f"    node {n.name!r} bl_idname={n.bl_idname}")
