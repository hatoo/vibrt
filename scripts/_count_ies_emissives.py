"""Find emissive meshes in ies_light."""
import bpy

print(f"=== meshes with emissive material ===")
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    for slot in obj.material_slots:
        m = slot.material
        if m is None or m.node_tree is None:
            continue
        for n in m.node_tree.nodes:
            if n.bl_idname in ("ShaderNodeEmission",):
                color = list(n.inputs["Color"].default_value)
                strength = n.inputs["Strength"].default_value
                print(f"  {obj.name!r} mat={m.name!r}: Emission color={color} strength={strength}")
            elif n.bl_idname == "ShaderNodeBsdfPrincipled":
                ec = n.inputs.get("Emission Color")
                es = n.inputs.get("Emission Strength")
                if ec and es:
                    cv = list(ec.default_value)
                    sv = float(es.default_value)
                    if sv > 0 and any(c > 0 for c in cv[:3]):
                        print(f"  {obj.name!r} mat={m.name!r}: Principled Emission color={cv} strength={sv}")

# All MESH objects positions
print(f"\n=== MESH objects ===")
for obj in bpy.data.objects:
    if obj.type == "MESH":
        d = obj.dimensions
        print(f"  {obj.name!r}: dim=({d.x:.2f},{d.y:.2f},{d.z:.2f}) "
              f"loc=({obj.location.x:.2f},{obj.location.y:.2f},{obj.location.z:.2f}) "
              f"mats=[{', '.join(s.material.name if s.material else '(none)' for s in obj.material_slots)}]")
