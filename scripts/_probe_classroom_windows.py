"""Investigate classroom's window materials. The mid-wall under-bright
gap is constant across bounce counts (NEE delivery issue, not indirect),
which points to shadow rays / sun rays not transmitting through windows
properly. Cycles uses portal sampling; vibrt relies on alpha_blend=0
+ shadow-ray passthrough. If the window MixShader doesn't match our
daylight portal heuristic the alpha_blend stays at 1 (opaque) and
shadow rays get blocked.
"""
from __future__ import annotations

import bpy

# Materials likely associated with windows
suspicion = ("window", "glass", "pane", "sun", "sky", "frame")

print("=== materials in classroom ===")
matched = []
for mat in bpy.data.materials:
    name_lc = mat.name.lower()
    if any(s in name_lc for s in suspicion):
        matched.append(mat)

for mat in matched:
    print(f"\n--- {mat.name} ---")
    if not mat.node_tree:
        print("  no node_tree")
        continue
    out = next((n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is None:
        print("  no output node")
        continue
    surf = out.inputs.get("Surface")
    if surf is None or not surf.is_linked:
        print("  output.Surface unlinked")
        continue
    src = surf.links[0].from_node
    print(f"  output.Surface ← {src.name} ({src.bl_idname})")
    # Walk one level
    if src.bl_idname == "ShaderNodeMixShader":
        n1 = src.inputs[1].links[0].from_node if src.inputs[1].is_linked else None
        n2 = src.inputs[2].links[0].from_node if src.inputs[2].is_linked else None
        fac = src.inputs[0]
        fac_kind = ("linked" if fac.is_linked else f"const {fac.default_value:.3f}")
        print(f"    Mix(shader1={n1.bl_idname if n1 else 'None'}/{n1.name if n1 else ''}, "
              f"shader2={n2.bl_idname if n2 else 'None'}/{n2.name if n2 else ''}, "
              f"fac={fac_kind})")
        for nn, label in ((n1, "shader1"), (n2, "shader2")):
            if nn is None:
                continue
            if nn.bl_idname == "ShaderNodeEmission":
                col = nn.inputs.get("Color")
                if col and col.is_linked:
                    src2 = col.links[0].from_node
                    print(f"    {label} Emission.Color ← {src2.bl_idname}/{src2.name}")
                else:
                    print(f"    {label} Emission.Color = const")
    elif src.bl_idname == "ShaderNodeBsdfTransparent":
        print("    pure Transparent")
    elif src.bl_idname == "ShaderNodeBsdfPrincipled":
        alpha = src.inputs.get("Alpha")
        if alpha is not None:
            ap = alpha.default_value if not alpha.is_linked else "linked"
            print(f"    Principled with Alpha = {ap}")

# Also: check actual scene objects whose names suggest windows
print("\n=== scene objects ===")
for obj in bpy.context.scene.objects:
    name_lc = obj.name.lower()
    if any(s in name_lc for s in suspicion):
        print(f"  {obj.name}: type={obj.type}, hide_render={obj.hide_render}, "
              f"vis_camera={obj.visible_camera}, vis_diffuse={obj.visible_diffuse}, "
              f"materials={[s.material.name if s.material else None for s in (obj.material_slots if obj.type=='MESH' else [])]}")
