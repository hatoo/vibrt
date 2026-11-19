"""Inspect the scene's compositor and any mist / world settings.

Run via: blender -b <scene>.blend --python scripts/_probe_compositor.py
"""
import bpy

scene = bpy.context.scene

# Compositor node tree lives at scene.compositing_node_group / scene.node_tree
# in different Blender versions; try both.
print("=== compositor ===")
for attr in ("node_tree", "compositing_node_group"):
    tree = getattr(scene, attr, None)
    if tree is not None:
        print(f"  scene.{attr} present: {tree.name if hasattr(tree, 'name') else type(tree).__name__}")

# Use scene.use_nodes (older API) but guarded against attribute issues.
try:
    use_n = scene.use_nodes
    print(f"  scene.use_nodes = {use_n}")
except Exception as e:
    print(f"  scene.use_nodes: error {e}")

# Try direct compositor node tree.
nt = None
for attr in ("node_tree", "compositing_node_group"):
    nt = getattr(scene, attr, None)
    if nt is not None and hasattr(nt, "nodes"):
        break

if nt is not None and hasattr(nt, "nodes"):
    print(f"\ncompositor nodes ({len(nt.nodes)}):")
    for n in nt.nodes:
        ins = []
        for s in n.inputs:
            if s.is_linked:
                ins.append(f"{s.name}<-{s.links[0].from_node.name}.{s.links[0].from_socket.name}")
            else:
                if hasattr(s, "default_value"):
                    try:
                        ins.append(f"{s.name}={s.default_value}")
                    except Exception:
                        pass
        bid = n.bl_idname
        extra = ""
        if bid == "CompositorNodeMixRGB" or bid == "CompositorNodeMix":
            for a in ("blend_type", "use_clamp", "factor_mode"):
                if hasattr(n, a):
                    extra += f" {a}={getattr(n, a)}"
        if bid == "CompositorNodeColorBalance":
            extra = f" lift={tuple(n.lift)} gamma={tuple(n.gamma)} gain={tuple(n.gain)}"
        if bid == "CompositorNodeHueSat":
            for a in ("hue", "saturation", "value"):
                if hasattr(n, a):
                    extra += f" {a}={getattr(n, a)}"
        if bid == "CompositorNodeRGBCurve":
            extra = f" mapping?"
        print(f"  {bid} '{n.name}'{extra}  mute={n.mute}")
        for line in ins:
            print(f"    in: {line}")
else:
    print("\n(no compositor node tree)")

# Mist pass settings.
print("\n=== world mist ===")
if scene.world is not None and hasattr(scene.world, "mist_settings"):
    ms = scene.world.mist_settings
    for a in ("use_mist", "intensity", "start", "depth", "height", "falloff"):
        if hasattr(ms, a):
            print(f"  world.mist_settings.{a} = {getattr(ms, a)}")

# View layer passes.
print("\n=== view layer passes ===")
vl = scene.view_layers[0]
for a in dir(vl):
    if a.startswith("use_pass_") and getattr(vl, a):
        print(f"  view_layer.{a} = True")
