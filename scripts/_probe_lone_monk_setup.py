"""Inspect lone_monk's world, lights, render settings, and look for
window / portal-like materials that might match the classroom pattern
(MixShader Transparent + Emission(TexSky)) we just fixed."""
import bpy

scene = bpy.context.scene
print(f"=== render settings ===")
cy = scene.cycles
print(f"  use_light_tree = {getattr(cy, 'use_light_tree', '?')}")
print(f"  diffuse_bounces = {cy.diffuse_bounces}, glossy = {cy.glossy_bounces}, "
      f"transmission = {cy.transmission_bounces}")
print(f"  sample_clamp_direct = {cy.sample_clamp_direct}, indirect = {cy.sample_clamp_indirect}")
print(f"  blur_glossy = {cy.blur_glossy}")

print(f"\n=== world '{scene.world.name}' ===")
if scene.world.use_nodes:
    for n in scene.world.node_tree.nodes:
        if n.bl_idname not in ("NodeReroute",):
            print(f"  {n.bl_idname}/{n.name}")
            for inp in n.inputs:
                if inp.is_linked:
                    src = inp.links[0].from_node
                    print(f"    in {inp.name} ← {src.bl_idname}/{src.name}")
                else:
                    try:
                        v = inp.default_value
                        if hasattr(v, "__iter__"):
                            v = tuple(round(float(x), 3) for x in v)
                        else:
                            v = round(float(v), 4) if isinstance(v, (int, float)) else v
                        print(f"    in {inp.name} = {v}")
                    except Exception:
                        pass

print(f"\n=== lights ===")
for obj in scene.objects:
    if obj.type == "LIGHT":
        light = obj.data
        print(f"  {obj.name}: type={light.type} color={tuple(round(float(c), 3) for c in light.color)} "
              f"strength={light.energy:.3f}")

print(f"\n=== Materials with MixShader patterns or TexSky ===")
for mat in bpy.data.materials:
    if mat.node_tree is None:
        continue
    has_mixshader = any(n.bl_idname == "ShaderNodeMixShader" for n in mat.node_tree.nodes)
    has_texsky = any(n.bl_idname == "ShaderNodeTexSky" for n in mat.node_tree.nodes)
    has_transparent = any(n.bl_idname == "ShaderNodeBsdfTransparent" for n in mat.node_tree.nodes)
    if has_mixshader and (has_texsky or has_transparent):
        print(f"  {mat.name}: mix={has_mixshader} sky={has_texsky} transparent={has_transparent}")

print(f"\n=== camera-hidden objects ===")
for obj in scene.objects:
    if obj.type == "MESH" and not obj.visible_camera:
        mats = [s.material.name if s.material else None for s in obj.material_slots]
        print(f"  {obj.name}: visible_camera=False, mats={mats}")
