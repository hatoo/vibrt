"""Show classroom's world setup + sun direction/strength.
"""
import bpy

scene = bpy.context.scene
world = scene.world
print(f"=== world {world.name!r} ===")
if world.node_tree:
    out = next((n for n in world.node_tree.nodes if n.type == "OUTPUT_WORLD"), None)
    if out:
        surf = out.inputs.get("Surface")
        if surf and surf.is_linked:
            src = surf.links[0].from_node
            print(f"  surface ← {src.bl_idname}/{src.name}")
            for inp in src.inputs:
                if inp.is_linked:
                    src2 = inp.links[0].from_node
                    print(f"    {inp.name} ← {src2.bl_idname}/{src2.name}")
                else:
                    try:
                        v = inp.default_value
                        if hasattr(v, "__iter__"):
                            v = tuple(round(float(x), 3) for x in v)
                        else:
                            v = round(float(v), 3) if isinstance(v, (int, float)) else v
                        print(f"    {inp.name} = {v}")
                    except Exception:
                        pass

print(f"\n=== sun lights ===")
for obj in scene.objects:
    if obj.type == "LIGHT" and obj.data.type == "SUN":
        light = obj.data
        # World direction = matrix_world @ -Z
        m = obj.matrix_world
        # Light shines along local -Z; "from" is +Z in world frame
        # but the visible light's shape matters
        z = m.col[2]
        forward = (-z[0], -z[1], -z[2])  # direction sun travels (light to surface)
        print(f"  {obj.name}: color={tuple(round(float(c), 3) for c in light.color)} "
              f"strength={round(float(light.energy), 3)} angle={round(float(light.angle), 4)}")
        print(f"    forward (where photons go) = {tuple(round(float(c), 4) for c in forward)}")
        print(f"    location = {tuple(round(float(c), 3) for c in obj.location)}")
        print(f"    visible_camera={obj.visible_camera}, visible_diffuse={obj.visible_diffuse}")

# Camera direction so we can place mid-wall geometry
cam = scene.camera
if cam:
    m = cam.matrix_world
    z = m.col[2]
    forward = (-z[0], -z[1], -z[2])
    print(f"\n=== camera ===")
    print(f"  loc = {tuple(round(float(c), 3) for c in cam.location)}")
    print(f"  forward (look dir) = {tuple(round(float(c), 4) for c in forward)}")

print(f"\n=== render settings ===")
cy = scene.cycles
print(f"  use_light_tree = {getattr(cy, 'use_light_tree', '?')}")
print(f"  light_sampling_threshold = {getattr(cy, 'light_sampling_threshold', '?')}")
print(f"  diffuse_bounces = {cy.diffuse_bounces}")
print(f"  glossy_bounces = {cy.glossy_bounces}")
print(f"  transmission_bounces = {cy.transmission_bounces}")
print(f"  sample_clamp_direct = {cy.sample_clamp_direct}")
print(f"  sample_clamp_indirect = {cy.sample_clamp_indirect}")
print(f"  blur_glossy = {cy.blur_glossy}")
