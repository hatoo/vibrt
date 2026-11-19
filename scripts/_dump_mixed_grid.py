"""Recompute pabellon's Mixed envmap grid in Python (mirroring
`build_mixed_envmap_grid` in scene_loader.rs) and save as PNG.

We then compare it against Cycles' panoramic equirect bake of the
exact same world to see if the discrepancy is in (a) the rasterisation
formula, (b) the Mapping rotation direction, or (c) the per-layer
strength order. All three would manifest differently."""
import bpy, os, sys
import math
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

from vibrt_blender.exporter import (
    prebake_sky_envmaps_for_world, _SKY_BAKE_CACHE,
    _sky_node_alone_cache_key, _detect_mix_world,
    _euler_xyz_to_matrix3,
)

scene = bpy.context.scene
world = scene.world

# Run the prebake so the Sky-alone cache is populated.
prebake_sky_envmaps_for_world(world, w=1024, h=512)

mix = _detect_mix_world(world)
if mix is None:
    raise SystemExit("not a Mixed world")
bg_a, bg_b, fac = mix

def get_layer(bg):
    color_sock = bg.inputs.get("Color")
    src = color_sock.links[0].from_node
    strength = float(bg.inputs["Strength"].default_value)
    rotation = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    if src.bl_idname == "ShaderNodeTexSky":
        key = _sky_node_alone_cache_key(world, src)
        rgb, w, h, _sun = _SKY_BAKE_CACHE[key]
        rgb = np.frombuffer(rgb, dtype=np.float32).reshape((h, w, 3))
        return ("equirect", rgb, w, h, np.array(rotation).reshape(3, 3),
                strength, "repeat")
    if src.bl_idname in ("ShaderNodeTexImage",):
        img = src.image
        # Force the image into linear space and grab pixels.
        prev_cs = img.colorspace_settings.name
        img.colorspace_settings.name = "Non-Color"
        try:
            arr = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
            img.pixels.foreach_get(arr)
        finally:
            img.colorspace_settings.name = prev_cs
        rgba = arr.reshape((img.size[1], img.size[0], 4))
        # Match scene_loader.rs raw layout: keep foreach_get's bottom-up
        # pixel order so y in [0, h] reads the same row Rust does.
        rgb = rgba[..., :3].copy()
        if prev_cs.lower().startswith("srgb"):
            # sRGB → linear (rough approximation of the proper EOTF)
            rgb = np.where(rgb <= 0.04045, rgb / 12.92,
                           ((rgb + 0.055) / 1.055) ** 2.4)
        proj = "flat" if src.projection == "FLAT" else "equirect"
        ext_map = {"REPEAT": "repeat", "EXTEND": "extend", "CLIP": "clip"}
        ext = ext_map.get(src.extension, "repeat")
        # Walk back to the Mapping rotation.
        rot = np.eye(3)
        v = src.inputs.get("Vector")
        if v and v.is_linked:
            mp = v.links[0].from_node
            if mp.bl_idname == "ShaderNodeMapping":
                rx, ry, rz = mp.inputs["Rotation"].default_value
                rot = np.array(_euler_xyz_to_matrix3(rx, ry, rz)).reshape(3, 3)
        return (proj, rgb, img.size[0], img.size[1], rot, strength, ext)
    raise RuntimeError("unsupported source")

la = get_layer(bg_a)
lb = get_layer(bg_b)
print(f"layer A: proj={la[0]} size={la[2]}x{la[3]} strength={la[5]} ext={la[6]}")
print(f"layer A rotation:\n{la[4]}")
print(f"layer B: proj={lb[0]} size={lb[2]}x{lb[3]} strength={lb[5]} ext={lb[6]}")
print(f"layer B rotation:\n{lb[4]}")
print(f"fac={fac}")

USE_TRANSPOSE = False
NEGATE_FLAT_DIR = False  # Rust scene_loader.rs samples (R · d) on bottom-up data


def sample(layer, dirs):
    proj, rgb, w, h, rot, strength, ext = layer
    use_dirs = dirs
    if proj == "flat" and NEGATE_FLAT_DIR:
        use_dirs = -dirs
    if USE_TRANSPOSE:
        rotated = use_dirs @ rot
    else:
        rotated = use_dirs @ rot.T
    if proj == "flat":
        u = rotated[:, 0]
        v = rotated[:, 1]
    else:
        z = np.clip(rotated[:, 2], -1, 1)
        theta = np.arccos(z)
        phi = np.arctan2(rotated[:, 1], rotated[:, 0])
        u = phi * (0.5 / math.pi)
        u = np.where(u < 0, u + 1, u)
        v = theta / math.pi
    # Bilinear sample.
    if ext == "repeat":
        u_w = u * w - 0.5
        v_w = v * h - 0.5
        x0 = np.floor(u_w).astype(int)
        y0 = np.floor(v_w).astype(int)
        dx = u_w - x0; dy = v_w - y0
        x0 = x0 % w; x1 = (x0 + 1) % w
        if proj == "equirect":
            y0c = np.clip(y0, 0, h - 1); y1c = np.clip(y0 + 1, 0, h - 1)
        else:
            y0c = y0 % h; y1c = (y0 + 1) % h
    else:
        u = np.clip(u, 0, 1); v = np.clip(v, 0, 1)
        u_w = u * w - 0.5; v_w = v * h - 0.5
        x0 = np.clip(np.floor(u_w).astype(int), 0, w - 1)
        y0c = np.clip(np.floor(v_w).astype(int), 0, h - 1)
        x1 = np.clip(x0 + 1, 0, w - 1); y1c = np.clip(y0c + 1, 0, h - 1)
        dx = u_w - x0; dy = v_w - y0c
    c00 = rgb[y0c, x0]; c10 = rgb[y0c, x1]
    c01 = rgb[y1c, x0]; c11 = rgb[y1c, x1]
    c0 = c00 * (1 - dx[:, None]) + c10 * dx[:, None]
    c1 = c01 * (1 - dx[:, None]) + c11 * dx[:, None]
    return (c0 * (1 - dy[:, None]) + c1 * dy[:, None]) * strength

target_w, target_h = 1024, 512
us = (np.arange(target_w) + 0.5) / target_w
vs = (np.arange(target_h) + 0.5) / target_h
uu, vv = np.meshgrid(us, vs)
phi = uu * 2 * math.pi
theta = vv * math.pi
dirs = np.stack([
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta),
], axis=-1).reshape(-1, 3)

a = sample(la, dirs)
b = sample(lb, dirs)
mixed = (a * (1 - fac) + b * fac).reshape(target_h, target_w, 3)

# Tone-map to LDR sRGB for visual comparison.
m = mixed.copy()
# Reinhard tone-map then sRGB encode
m = m / (1 + m)
m = np.where(m <= 0.0031308, m * 12.92, 1.055 * np.power(m, 1/2.4) - 0.055)
m = np.clip(m * 255, 0, 255).astype(np.uint8)

out = os.path.join(HERE, "test_scenes", "pabellon", "_mixed_grid.png")
img = bpy.data.images.new("_grid", target_w, target_h, alpha=False)
out_arr = np.dstack([m.astype(np.float32) / 255.0,
                     np.ones((target_h, target_w), dtype=np.float32)])
# Blender stores pixels bottom-up; flip back.
out_arr = out_arr[::-1].reshape(-1)
img.pixels.foreach_set(out_arr)
img.filepath_raw = out
img.file_format = "PNG"
img.save()
print(f"wrote {out}")
print(f"mixed grid mean rgb (linear) = {tuple(round(float(x), 4) for x in mixed.mean(axis=(0,1)))}")
print(f"layer A alone mean = {tuple(round(float(x), 4) for x in (a.reshape(target_h, target_w, 3) * (1-fac)).mean(axis=(0,1)))}")
print(f"layer B alone mean = {tuple(round(float(x), 4) for x in (b.reshape(target_h, target_w, 3) * fac).mean(axis=(0,1)))}")
