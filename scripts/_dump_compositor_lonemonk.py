"""Dump every link in the lone_monk compositor so we can see exactly what
the Mix nodes do (colour B input, factor source, blend mode etc.)."""
import bpy

scene = bpy.context.scene
nt = scene.compositing_node_group or scene.node_tree
print(f"compositor: {len(nt.nodes)} nodes")
for n in nt.nodes:
    print(f"\n[{n.bl_idname}] '{n.name}' mute={n.mute}")
    for s in n.inputs:
        if s.is_linked:
            link = s.links[0]
            print(f"  in {s.name:20s} <- {link.from_node.name}.{link.from_socket.name}")
        else:
            try:
                dv = s.default_value
                if hasattr(dv, '__len__'):
                    val = "(" + ", ".join(f"{v:.3f}" for v in dv) + ")"
                else:
                    val = str(dv)
                print(f"  in {s.name:20s} = {val}")
            except Exception:
                print(f"  in {s.name:20s} (no default)")
    if n.bl_idname in ("CompositorNodeMixRGB",):
        print(f"  blend_type: {n.blend_type}, use_clamp: {n.use_clamp}")
    if n.bl_idname == "ShaderNodeMix":
        print(f"  data_type: {n.data_type}, blend_type: {n.blend_type}, "
              f"clamp_factor: {getattr(n, 'clamp_factor', '?')}")
    if n.bl_idname == "CompositorNodeGlare":
        for a in ("glare_type", "size", "iterations", "threshold", "fade",
                  "color_modulation", "mix"):
            if hasattr(n, a):
                print(f"  {a} = {getattr(n, a)}")
    if n.bl_idname == "ShaderNodeValToRGB":
        ramp = n.color_ramp
        print(f"  color_mode={ramp.color_mode}, interpolation={ramp.interpolation}")
        for i, e in enumerate(ramp.elements):
            c = e.color
            print(f"  stop[{i}]: pos={e.position:.4f} col=({c[0]:.4f},{c[1]:.4f},{c[2]:.4f},{c[3]:.4f})")
    if n.bl_idname == "ShaderNodeVectorMath":
        print(f"  operation = {n.operation}")
    if n.bl_idname == "CompositorNodeExposure":
        print(f"  (multiplies image by 2^Exposure)")
    if n.bl_idname == "CompositorNodeDilateErode":
        print(f"  mode = {getattr(n, 'mode', '?')}, distance = {getattr(n, 'distance', '?')}")
