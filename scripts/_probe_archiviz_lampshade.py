"""Inspect the 'Lampshade' and 'Lamp Gubi' materials in flat-archiviz to see
whether the brightness in the Cycles render comes from emission baked into the
shade itself rather than the Point lights."""
import bpy

for mat_name in ("Lampshade", "Lamp Gubi"):
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        print(f"-- {mat_name}: NOT FOUND --")
        continue
    print(f"=== {mat_name} ===")
    if not mat.use_nodes:
        print("  (no nodes)")
        continue
    nt = mat.node_tree
    for n in nt.nodes:
        line = f"  {n.bl_idname:35s} '{n.name}'"
        if n.bl_idname == 'ShaderNodeBsdfPrincipled':
            em_sock = n.inputs.get("Emission") or n.inputs.get("Emission Color")
            es_sock = n.inputs.get("Emission Strength")
            ec = list(em_sock.default_value)[:3] if em_sock and not em_sock.is_linked else "(linked)"
            es = es_sock.default_value if es_sock and not es_sock.is_linked else "(linked)"
            line += f"  emission_color={ec} emission_strength={es}"
        elif n.bl_idname == 'ShaderNodeEmission':
            cs = n.inputs.get("Color")
            ss = n.inputs.get("Strength")
            cv = list(cs.default_value)[:3] if cs and not cs.is_linked else "(linked)"
            sv = ss.default_value if ss and not ss.is_linked else "(linked)"
            line += f"  color={cv} strength={sv}"
        elif hasattr(n, 'image') and getattr(n, 'image', None):
            line += f"  image='{n.image.name}'"
        print(line)
    print("  links:")
    for L in nt.links:
        print(f"    {L.from_node.name}.{L.from_socket.name} -> {L.to_node.name}.{L.to_socket.name}")
    print()
