"""Audit every TexImage in lone_monk's materials: dump filename,
Blender colorspace setting, where it lands in the material (base_color
vs roughness vs normal vs ...), and whether the wired socket implies
sRGB or linear semantics. Used to triage whether vibrt is reading any
texture at the wrong colorspace, which would explain the lit-wall dim.
"""
import bpy

scene = bpy.context.scene

def socket_role(sock):
    """Best-effort guess of what the wired socket is used as."""
    name = sock.name
    if name in ("Base Color", "Color", "Color 1", "Color 2", "A", "B"):
        return "color"  # sRGB-encoded by default
    if name in ("Normal", "Tangent"):
        return "normal"  # linear
    if name in ("Roughness", "Metallic", "Specular", "Specular IOR Level",
                 "Transmission", "Alpha", "Coat Roughness", "Coat IOR",
                 "Coat Weight", "Sheen Roughness", "Sheen Weight",
                 "Strength", "Factor", "Fac", "Hue", "Saturation", "Value",
                 "Translucent", "Anisotropic"):
        return "scalar"  # linear data, single channel
    if name in ("Vector", "UV"):
        return "vector"
    return f"other:{name}"


def walk_image_users(node_tree, mat_name, path=""):
    """Yield (image, role, mat, node_name, socket_path) tuples for every
    TexImage in a node tree (recursing into NodeGroups)."""
    for n in node_tree.nodes:
        if n.bl_idname == "ShaderNodeTexImage" and n.image is not None:
            users = []
            for out_sock in n.outputs:
                for link in out_sock.links:
                    role = socket_role(link.to_socket)
                    target_node = link.to_node
                    users.append(
                        f"{out_sock.name}→{target_node.bl_idname}.{link.to_socket.name} ({role})"
                    )
            yield (n.image, n.name, users, mat_name)
        elif n.bl_idname == "ShaderNodeGroup" and n.node_tree is not None:
            yield from walk_image_users(n.node_tree, mat_name, path + f">{n.name}")


# Collect material → list of (image, name, users)
all_users = []
seen = set()
for mat in bpy.data.materials:
    if mat.node_tree is None:
        continue
    for img, node_name, users, mname in walk_image_users(mat.node_tree, mat.name):
        key = (img.name, mname)
        if key in seen:
            continue
        seen.add(key)
        all_users.append((mat, img, node_name, users))

print(f"=== {len(all_users)} TexImage usages across materials ===")
print(f"{'material':40s} {'image':40s} {'colorspace':12s} {'users'}")
for mat, img, node_name, users in all_users:
    cs = img.colorspace_settings.name
    users_str = " | ".join(users) if users else "(unconnected)"
    flag = ""
    # Heuristic: if image is wired to a color slot but colorspace is linear
    # (or wired to a scalar/normal slot but colorspace is sRGB), that's suspicious.
    color_users = [u for u in users if "(color)" in u]
    scalar_users = [u for u in users if "(scalar)" in u or "(normal)" in u]
    if color_users and cs.lower() not in ("srgb", "filmic srgb"):
        flag = " ⚠ COLOR socket on non-sRGB texture"
    if scalar_users and cs.lower().startswith("srgb"):
        flag = " ⚠ SCALAR/NORMAL socket on sRGB texture"
    print(f"{mat.name:40s} {img.name:40s} {cs:12s} {users_str}{flag}")
