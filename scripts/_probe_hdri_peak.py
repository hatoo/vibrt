"""Read pabellon's HDRI image directly and report its peak luminance,
to see whether the underlying source has bright HDR pixels (just lost
during the bake) or whether the dimness is inherent."""
import bpy, os, numpy as np

# Walk the world's node tree for any TexImage feeding a Background.
world = bpy.context.scene.world
print(f"world: {world.name}")

def walk(node, found, depth=0):
    if depth > 8:
        return
    for inp in node.inputs:
        if not inp.is_linked:
            continue
        src = inp.links[0].from_node
        if src.bl_idname == "ShaderNodeTexImage" and src.image is not None:
            found.append(src)
        walk(src, found, depth + 1)

found = []
out = next(n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeOutputWorld")
walk(out, found)
print(f"found {len(found)} TexImage nodes")
for n in found:
    img = n.image
    print(f"\nimage '{img.name}': size={tuple(img.size)} colorspace={img.colorspace_settings.name}")
    w, h = int(img.size[0]), int(img.size[1])
    if w == 0 or h == 0:
        img.update()
        w, h = int(img.size[0]), int(img.size[1])
    if w == 0 or h == 0:
        continue
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    rgba = flat.reshape((h, w, 4))
    rgb = rgba[..., :3]
    # If sRGB, linearize
    is_srgb = img.colorspace_settings.name.lower().startswith("srgb")
    if is_srgb:
        rgb = np.where(rgb <= 0.0404482362771082,
                       rgb / 12.92,
                       ((rgb + 0.055) / 1.055) ** 2.4)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    print(f"  pixel max RGB: {tuple(round(float(rgb[..., c].max()), 3) for c in range(3))}")
    print(f"  pixel mean RGB: {tuple(round(float(rgb[..., c].mean()), 3) for c in range(3))}")
    print(f"  luminance: max={float(lum.max()):.3f} p99={float(np.percentile(lum, 99)):.3f} median={float(np.percentile(lum, 50)):.3f}")
    # Find brightest pixel position
    idx = np.argmax(lum)
    y, x = idx // w, idx % w
    u = (x + 0.5) / w
    v = (y + 0.5) / h  # bottom-up
    print(f"  brightest pixel at (x,y)=({x},{y}) = uv=({u:.3f}, {v:.3f}) [bottom-up]")
