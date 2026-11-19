"""Probe object-level visibility flags + the 'Material' shader graph
on `Plane` to see whether Cycles treats it as camera-visible at all,
and whether the material is something invisible / mask-only that vibrt
mishandles."""
import bpy

obj = bpy.data.objects.get("Plane")
if obj is None:
    raise SystemExit("no 'Plane'")
print(f"=== object 'Plane' visibility ===")
print(f"  hide_render={obj.hide_render} hide_viewport={obj.hide_viewport} hide_select={obj.hide_select}")
print(f"  visible_camera={obj.visible_camera} visible_diffuse={obj.visible_diffuse} "
      f"visible_glossy={obj.visible_glossy} visible_transmission={obj.visible_transmission} "
      f"visible_volume_scatter={obj.visible_volume_scatter} visible_shadow={obj.visible_shadow}")
# Cycles object props
cy = getattr(obj, "cycles", None)
if cy is not None:
    print(f"  cycles.is_shadow_catcher={getattr(cy, 'is_shadow_catcher', None)}")
    print(f"  cycles.is_holdout={getattr(cy, 'is_holdout', None)}")
print(f"  display_type={obj.display_type}")
print()

# Walk the 'Material' graph.
def walk(n, depth=0, seen=None):
    if seen is None: seen = set()
    if n.name in seen: return
    seen = seen | {n.name}
    indent = "  " * depth
    extra = ""
    if n.bl_idname == "ShaderNodeBsdfPrincipled":
        ar = n.inputs.get("Alpha")
        if ar:
            extra = f" Alpha={ar.default_value if not ar.is_linked else 'linked'}"
    if n.bl_idname == "ShaderNodeBsdfTransparent":
        extra = " <transparent>"
    if n.bl_idname == "ShaderNodeMixShader":
        extra = " <mix>"
    if n.bl_idname == "ShaderNodeHoldout":
        extra = " <HOLDOUT>"
    print(f"{indent}{n.bl_idname} '{n.name}'{extra}")
    for inp in n.inputs:
        if inp.is_linked:
            link = inp.links[0]
            print(f"{indent}  in {inp.name}: {link.from_node.bl_idname}.{link.from_socket.name}")
            walk(link.from_node, depth + 2, seen)

mat = bpy.data.materials.get("Material")
if mat is None:
    raise SystemExit("no 'Material'")
print(f"=== material 'Material' (use_nodes={mat.use_nodes}) ===")
print(f"  blend_method={mat.blend_method} shadow_method={getattr(mat, 'shadow_method', None)}")
print(f"  diffuse_color={tuple(round(c, 3) for c in mat.diffuse_color)}")
out = next((n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
if out:
    walk(out)
