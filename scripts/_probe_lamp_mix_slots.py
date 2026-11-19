"""Identify which BSDF is wired to MixShader slot 1 vs slot 2 for Lamp Gubi."""
import bpy

mat = bpy.data.materials["Lamp Gubi"]
nt = mat.node_tree
mix = next(n for n in nt.nodes if n.bl_idname == 'ShaderNodeMixShader')
print(f"Mix Shader '{mix.name}' inputs:")
for i, inp in enumerate(mix.inputs):
    print(f"  inputs[{i}] name={inp.name!r} type={inp.type}")
    for L in inp.links:
        print(f"    <- {L.from_node.bl_idname} '{L.from_node.name}' .{L.from_socket.name}")
