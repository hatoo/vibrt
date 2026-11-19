"""Dump white_lotus_leafs ColorRamp stops + a few sample colors."""
import bpy

mat = bpy.data.materials.get("white_lotus_leafs")
for n in mat.node_tree.nodes:
    if n.bl_idname != "ShaderNodeValToRGB":
        continue
    print(f"=== ColorRamp '{n.name}' ===")
    cr = n.color_ramp
    print(f"  interpolation={cr.interpolation}, color_mode={cr.color_mode}")
    for el in cr.elements:
        print(f"  pos={el.position:.3f} col={tuple(el.color[:3])} alpha={el.color[3]:.3f}")
    # Sample at a few points
    print("  samples:")
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        c = cr.evaluate(t)
        print(f"    t={t:.2f} -> {tuple(round(x,3) for x in c[:3])}")
    # Mean
    n_s = 64
    acc = [0.0, 0.0, 0.0]
    for i in range(n_s):
        t = (i + 0.5) / n_s
        c = cr.evaluate(t)
        for k in range(3):
            acc[k] += c[k]
    mean = [v / n_s for v in acc]
    print(f"  mean over [0,1]: {tuple(round(x,3) for x in mean)}")

# also dump leafs (the autumn tree material) which shares the same idiom
print()
mat = bpy.data.materials.get("leafs")
if mat:
    for n in mat.node_tree.nodes:
        if n.bl_idname != "ShaderNodeValToRGB":
            continue
        print(f"=== leafs ColorRamp '{n.name}' ===")
        cr = n.color_ramp
        for el in cr.elements:
            print(f"  pos={el.position:.3f} col={tuple(el.color[:3])} alpha={el.color[3]:.3f}")
        n_s = 64
        acc = [0.0, 0.0, 0.0]
        for i in range(n_s):
            t = (i + 0.5) / n_s
            c = cr.evaluate(t)
            for k in range(3):
                acc[k] += c[k]
        mean = [v / n_s for v in acc]
        print(f"  mean over [0,1]: {tuple(round(x,3) for x in mean)}")
