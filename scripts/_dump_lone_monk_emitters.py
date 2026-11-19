"""Walk all lone_monk materials and dump emission paths (Emission shader,
Principled BSDF emission, MixShader chains) with their strength and color.
Goal: understand which emissive materials exist, what Le they should yield,
and whether vibrt's exporter promotes them to area_rect lights.
"""
import bpy

scene = bpy.context.scene


def trace_emission(node, depth=0):
    """Walk back from the input recursively, return emission contributions."""
    if node.bl_idname == "ShaderNodeEmission":
        col_socket = node.inputs.get("Color")
        str_socket = node.inputs.get("Strength")
        col = "linked" if col_socket.is_linked else tuple(round(float(x), 3) for x in col_socket.default_value)
        s = "linked" if str_socket.is_linked else round(float(str_socket.default_value), 3)
        return [("Emission", col, s)]
    if node.bl_idname == "ShaderNodeBsdfPrincipled":
        ec_socket = node.inputs.get("Emission Color")
        es_socket = node.inputs.get("Emission Strength")
        col = "linked" if (ec_socket and ec_socket.is_linked) else tuple(round(float(x), 3) for x in ec_socket.default_value) if ec_socket else None
        s = "linked" if (es_socket and es_socket.is_linked) else round(float(es_socket.default_value), 3) if es_socket else 0
        if (isinstance(s, (int, float)) and s > 0) or s == "linked":
            return [(f"PrincipledEmission", col, s)]
        return []
    if node.bl_idname == "ShaderNodeMixShader":
        out = []
        for inp_idx in (1, 2):
            sock = node.inputs[inp_idx]
            if sock.is_linked:
                src = sock.links[0].from_node
                out.extend(trace_emission(src, depth + 1))
        return out
    if node.bl_idname == "ShaderNodeAddShader":
        out = []
        for inp_idx in (0, 1):
            sock = node.inputs[inp_idx]
            if sock.is_linked:
                src = sock.links[0].from_node
                out.extend(trace_emission(src, depth + 1))
        return out
    return []


print(f"=== Materials with emission ===")
emit_mats = []
for mat in bpy.data.materials:
    if mat.node_tree is None:
        continue
    out_node = next((n for n in mat.node_tree.nodes
                     if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output), None)
    if out_node is None:
        continue
    surf_socket = out_node.inputs.get("Surface")
    if surf_socket is None or not surf_socket.is_linked:
        continue
    src = surf_socket.links[0].from_node
    em = trace_emission(src)
    if em:
        emit_mats.append((mat.name, em))
        print(f"  '{mat.name}': {em}")

print(f"\n  Total emissive materials: {len(emit_mats)}")

# For each emissive material, find which mesh objects use it
print(f"\n=== Mesh objects per emissive material ===")
for mat_name, _ in emit_mats:
    objs = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        for ms in obj.material_slots:
            if ms.material is not None and ms.material.name == mat_name:
                # Compute mesh polygon count for this material
                mat_idx = list(obj.material_slots).index(ms)
                tri_count = sum(1 for p in obj.data.polygons if p.material_index == mat_idx)
                objs.append((obj.name, tri_count, list(obj.location), list(obj.dimensions)))
                break
    print(f"  '{mat_name}': {len(objs)} obj(s)")
    for n, t, loc, dim in objs[:5]:
        print(f"    {n} ({t} tris) loc={tuple(round(x, 2) for x in loc)} dim={tuple(round(x, 2) for x in dim)}")
    if len(objs) > 5:
        print(f"    ... +{len(objs) - 5} more")
