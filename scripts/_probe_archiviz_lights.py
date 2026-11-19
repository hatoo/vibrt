"""Inspect Point lights in flat-archiviz: their UI energy, emission node graph,
and whether a TexIES is wired to Strength. We need to know whether artists
relied on IES to do the heavy lifting (light.energy=1, IES does ×1000 in cone)
or whether IES is just a cosmetic falloff (light.energy already big enough)."""
import bpy

scene = bpy.context.scene
print("=== lights with TexIES on Emission.Strength ===")
for obj in scene.objects:
    if obj.type != 'LIGHT':
        continue
    light = obj.data
    if not light.use_nodes or light.node_tree is None:
        continue
    out = next((n for n in light.node_tree.nodes
                if n.bl_idname == "ShaderNodeOutputLight"), None)
    if out is None or not out.inputs["Surface"].is_linked:
        continue
    src = out.inputs["Surface"].links[0].from_node
    if src.bl_idname != "ShaderNodeEmission":
        continue
    s_sock = src.inputs.get("Strength")
    c_sock = src.inputs.get("Color")
    if s_sock is None or not s_sock.is_linked:
        continue
    up = s_sock.links[0].from_node
    if up.bl_idname != "ShaderNodeTexIES":
        continue

    ies_node = up
    ies_mode = getattr(ies_node, "mode", "?")
    ies_filepath = getattr(ies_node, "filepath", "")
    ies_text_name = ies_node.ies.name if (ies_mode == 'INTERNAL' and ies_node.ies) else ""
    ies_strength_input = ies_node.inputs.get("Strength")
    ies_strength = (ies_strength_input.default_value
                    if ies_strength_input is not None and not ies_strength_input.is_linked
                    else "(linked)")
    color = (
        c_sock.default_value if c_sock is not None and not c_sock.is_linked else "(linked)"
    )
    if isinstance(color, tuple) or hasattr(color, "__len__"):
        color = list(color)
    print(f"  {obj.name:30s}  type={light.type:10s}  energy={light.energy:.3f}")
    print(f"    light.color = {tuple(light.color)}")
    print(f"    Emission.Strength → TexIES, mode={ies_mode}, ies_strength={ies_strength}")
    print(f"    TexIES filepath = {ies_filepath!r}, internal_text = {ies_text_name!r}")
    print(f"    Emission.Color = {color}")

print()
print("=== all Point lights summary ===")
for obj in scene.objects:
    if obj.type != 'LIGHT' or obj.data.type != 'POINT':
        continue
    print(f"  {obj.name:30s}  energy={obj.data.energy:8.3f}  loc=({obj.location.x:+.2f},{obj.location.y:+.2f},{obj.location.z:+.2f})")
