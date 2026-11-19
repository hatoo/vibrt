"""Render lone_monk via Cycles, but swap the world's procedural Sky
Texture for an Environment Texture pointing at vibrt's prebaked EXR
of the same Nishita output. If the result matches Cycles' direct
Nishita render, vibrt's bake is faithful and the lone_monk gap is in
vibrt's renderer; if it's brighter (matching the original Cycles
preview), the bake is missing flux that Cycles' direct Sky Texture
evaluation captures.

Usage:
    blender -b lone_monk.blend --python this.py -- <bake.exr> <out.png>

The bake EXR must already exist (regenerate via the vibrt addon's
update() hook, e.g., one render or `_dump_lone_monk_sky_bake.py`).
"""
import bpy
import sys
import os

argv = sys.argv
sep = argv.index("--") if "--" in argv else None
extra = argv[sep + 1:] if sep is not None else []
if len(extra) < 2:
    print("usage: ... -- <bake.exr> <out.png>")
    sys.exit(1)
bake_exr, out_png = extra[0], extra[1]

scene = bpy.context.scene
w = scene.world
nt = w.node_tree

# Locate the World Output and current Background source.
out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputWorld")
bg = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBackground")

# Replace Sky Texture → Environment Texture pointing at the bake.
# Drop the existing Color link (Sky Texture → Background.Color).
for lk in list(nt.links):
    if lk.to_node is bg and lk.to_socket.name == "Color":
        nt.links.remove(lk)

env = nt.nodes.new("ShaderNodeTexEnvironment")
img = bpy.data.images.load(bake_exr)
img.colorspace_settings.name = "Non-Color"  # linear
env.image = img
env.projection = "EQUIRECTANGULAR"
nt.links.new(env.outputs["Color"], bg.inputs["Color"])

# Match the bake camera's orientation: vibrt's `_bake_sky_world_to_pixels`
# rotates the bake camera by (-π/2, 0, 0) so its +Y_local lands at world
# -Z, putting buffer y=0 at the world +Z direction. The downstream kernel
# then samples the resulting equirect with theta from world +Z. Cycles'
# default Environment Texture equirect convention is the same, so no
# Mapping node is needed for a direct round-trip.

# Render at the scene's existing settings, just to a different output path.
# Lower spp than the canonical 4096 — we just need enough to see the gap.
scene.cycles.samples = 4096
scene.cycles.use_adaptive_sampling = True
scene.cycles.use_denoising = True
scene.render.filepath = out_png
scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
print(f"[bake-world test] wrote {out_png}")
