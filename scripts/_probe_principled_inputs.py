"""Print every input name on Lamp Gubi's Principled BSDF to see whether
Blender 5.x exposes both 'Emission' (legacy alias) and 'Emission Color'."""
import bpy

mat = bpy.data.materials["Lamp Gubi"]
nt = mat.node_tree
princ = next(n for n in nt.nodes if n.bl_idname == 'ShaderNodeBsdfPrincipled')
print(f"=== {princ.bl_idname} '{princ.name}' inputs ({len(princ.inputs)}) ===")
for s in princ.inputs:
    linked = "LINKED" if s.is_linked else "free"
    dv = "?"
    try:
        if hasattr(s, 'default_value'):
            dv = list(s.default_value)[:4] if hasattr(s.default_value, '__len__') else s.default_value
    except Exception:
        pass
    print(f"  {s.name!r:40s} type={s.type:8s} {linked:6s} default={dv}")
