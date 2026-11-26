"""Sanity-check: build a world that paints +X red, +Y green, +Z blue
based on TexCoord.Generated, bake it via the addon, and report where
each colour lands in the equirect.
"""
import os, sys, math
import bpy
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender.exporter as exporter
from vibrt_blender.export_context import ExportContext
export_context = ExportContext()

# Build a simple world: Generated coordinate -> Color via Separate XYZ -> ColorRamp.
# We use a Math/Vector graph: red = max(0, x_gen), green = max(0, y_gen), blue = max(0, z_gen).
# Generated in worlds = ray direction in world space. Equivalent of a debug
# "world axis tint".
world = bpy.data.worlds.new("__rgb_world")
world.use_nodes = True
nt = world.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputWorld")
bg = nt.nodes.new("ShaderNodeBackground")
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
tc = nt.nodes.new("ShaderNodeTexCoord")
combine = nt.nodes.new("ShaderNodeCombineXYZ")
nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
nt.links.new(sep.outputs["X"], combine.inputs["X"])
nt.links.new(sep.outputs["Y"], combine.inputs["Y"])
nt.links.new(sep.outputs["Z"], combine.inputs["Z"])
nt.links.new(combine.outputs["Vector"], bg.inputs["Color"])
nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

# Bake at 4×2 so we can inspect a few specific directions.
export_context.sky_maps.clear()
exporter._prebake_world_full(world, w=512, h=256, context=export_context)

key = exporter._world_generic_cache_key(world)
cached = export_context.sky_maps[key]
rgb, bw, bh, _ = cached
# Layout: rgb has shape (bh, bw, 3) and was already mirrored in `_bake_sky_world_to_pixels`.
# `image.pixels.foreach_get` is bottom-up — `arr[0, :]` is the BOTTOM of the rendered EXR.

print(f"bake: {bw}x{bh}")
# Probe specific (theta, phi) directions that the KERNEL reads:
#   y = (int)(theta/pi * bh), x = (int)(phi/(2pi) * bw)
def kernel_sample(theta_rad, phi_rad):
    u = (phi_rad / (2 * math.pi)) % 1.0
    v = theta_rad / math.pi
    x = min(int(u * bw), bw - 1)
    y = min(int(v * bh), bh - 1)
    return rgb[y, x, :]

# Cardinal directions in world space:
tests = [
    ("+X (east)",  (math.pi/2, 0.0)),
    ("+Y (north)", (math.pi/2, math.pi/2)),
    ("-X",         (math.pi/2, math.pi)),
    ("-Y",         (math.pi/2, -math.pi/2)),
    ("+Z (zenith)", (0.001, 0.0)),
    ("-Z (nadir)",  (math.pi - 0.001, 0.0)),
]
for name, (th, ph) in tests:
    c = kernel_sample(th, ph)
    print(f"  kernel-sample {name}: rgb={tuple(round(float(x),3) for x in c)}")
print()
print("Expected: +X -> red(1,0,0), +Y -> green(0,1,0), +Z -> blue(0,0,1)")
print()

# Try fix candidates and report which gives correct +X→red, +Y→green.
def expect(name, expected_rgb, c):
    err = sum((float(c[i]) - expected_rgb[i])**2 for i in range(3))
    return err

print("--- Try azimuth shifts (no mirror) ---")
unmirrored = rgb[:, ::-1, :].copy()  # undo the existing mirror
for shift in (-bw//4, 0, bw//4, bw//2):
    rolled = np.roll(unmirrored, shift, axis=1)
    def s(th, ph):
        u = (ph / (2*math.pi)) % 1.0
        v = th / math.pi
        x = min(int(u * bw), bw - 1)
        y = min(int(v * bh), bh - 1)
        return rolled[y, x, :]
    px = s(math.pi/2, 0.0)
    py = s(math.pi/2, math.pi/2)
    print(f"  shift={shift:5d}: +X={tuple(round(float(x),3) for x in px)} +Y={tuple(round(float(x),3) for x in py)}")
print()
print("--- Try azimuth shifts WITH mirror (current behavior) ---")
for shift in (-bw//4, 0, bw//4, bw//2):
    rolled = np.roll(rgb, shift, axis=1)
    def s(th, ph):
        u = (ph / (2*math.pi)) % 1.0
        v = th / math.pi
        x = min(int(u * bw), bw - 1)
        y = min(int(v * bh), bh - 1)
        return rolled[y, x, :]
    px = s(math.pi/2, 0.0)
    py = s(math.pi/2, math.pi/2)
    print(f"  shift={shift:5d}: +X={tuple(round(float(x),3) for x in px)} +Y={tuple(round(float(x),3) for x in py)}")
