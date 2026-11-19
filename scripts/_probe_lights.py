"""Dump every light in the scene + the world shader summary.

Run via: blender -b <scene>.blend --python scripts/_probe_lights.py
"""

import bpy

scene = bpy.context.scene

print("=" * 70)
print(f"scene: {scene.name}")
print(f"world: {scene.world.name if scene.world else '(none)'}")
print("=" * 70)


def _node_summary(node, depth=0):
    indent = "  " * depth
    bid = node.bl_idname
    extra = ""
    if bid == "ShaderNodeRGB":
        c = node.outputs[0].default_value
        extra = f" RGB=({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}, {c[3]:.4f})"
    elif bid == "ShaderNodeValue":
        extra = f" value={node.outputs[0].default_value:.6f}"
    elif bid == "ShaderNodeBlackbody":
        t = node.inputs.get("Temperature")
        if t is not None:
            extra = f" temperature={t.default_value:.2f}K"
    elif bid == "ShaderNodeEmission":
        col = node.inputs.get("Color")
        s = node.inputs.get("Strength")
        if col is not None and not col.is_linked:
            c = col.default_value
            extra += f" Color=({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})"
        elif col is not None:
            extra += f" Color=<linked from {col.links[0].from_node.bl_idname}.{col.links[0].from_socket.name}>"
        if s is not None and not s.is_linked:
            extra += f" Strength={s.default_value:.4f}"
        elif s is not None:
            extra += f" Strength=<linked from {s.links[0].from_node.bl_idname}.{s.links[0].from_socket.name}>"
    elif bid == "ShaderNodeBackground":
        col = node.inputs.get("Color")
        s = node.inputs.get("Strength")
        if col is not None and not col.is_linked:
            c = col.default_value
            extra += f" Color=({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})"
        elif col is not None:
            extra += f" Color=<linked from {col.links[0].from_node.bl_idname}.{col.links[0].from_socket.name}>"
        if s is not None and not s.is_linked:
            extra += f" Strength={s.default_value:.4f}"
        elif s is not None:
            extra += f" Strength=<linked from {s.links[0].from_node.bl_idname}.{s.links[0].from_socket.name}>"
    elif bid == "ShaderNodeTexSky":
        # Properties differ between the two sky models; just print common ones.
        attrs = []
        for a in ("sky_type", "sun_size", "sun_intensity", "sun_elevation",
                  "sun_rotation", "altitude", "air_density", "dust_density",
                  "ozone_density", "turbidity", "ground_albedo"):
            if hasattr(node, a):
                v = getattr(node, a)
                attrs.append(f"{a}={v}")
        extra = "  " + ", ".join(attrs)
    print(f"{indent}- {node.bl_idname} '{node.name}'{extra}")


def _walk_node_tree(tree, label):
    if tree is None:
        return
    print(f"\n  ({label} node tree, {len(tree.nodes)} nodes)")
    for n in tree.nodes:
        _node_summary(n, depth=2)
    print(f"  links:")
    for link in tree.links:
        print(f"    {link.from_node.name}.{link.from_socket.name} → {link.to_node.name}.{link.to_socket.name}")


# === World ===
print()
print("--- WORLD ---")
if scene.world is not None:
    w = scene.world
    print(f"world.use_nodes = {w.use_nodes}")
    print(f"world.color = {tuple(w.color)}")
    if w.use_nodes:
        _walk_node_tree(w.node_tree, "world")

# === Scene lights ===
print()
print("--- LIGHTS ---")
for obj in sorted(scene.objects, key=lambda o: o.name):
    if obj.type != "LIGHT":
        continue
    li = obj.data
    print(f"\n[{obj.name}] type={li.type}")
    print(f"  color = {tuple(li.color)}")
    print(f"  energy = {li.energy}")
    print(f"  use_nodes = {li.use_nodes}")
    if li.type == "SUN":
        print(f"  angle = {li.angle}")
    elif li.type == "POINT":
        print(f"  shadow_soft_size = {li.shadow_soft_size}")
    elif li.type == "AREA":
        print(f"  shape = {li.shape}, size = {li.size}, size_y = {getattr(li, 'size_y', None)}")
    elif li.type == "SPOT":
        print(f"  spot_size = {li.spot_size}, spot_blend = {li.spot_blend}, shadow_soft_size = {li.shadow_soft_size}")
    if li.use_nodes and li.node_tree is not None:
        _walk_node_tree(li.node_tree, f"light {obj.name}")

# === Cycles render settings that affect colour ===
print()
print("--- RENDER / COLOR MGMT ---")
print(f"render.engine = {scene.render.engine}")
print(f"view_settings.view_transform = {scene.view_settings.view_transform}")
print(f"view_settings.look = {scene.view_settings.look}")
print(f"view_settings.exposure = {scene.view_settings.exposure}")
print(f"view_settings.gamma = {scene.view_settings.gamma}")
print(f"display_settings.display_device = {scene.display_settings.display_device}")
