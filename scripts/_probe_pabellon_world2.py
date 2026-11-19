"""Deeper world-graph probe for pabellon. Specifically:
  - TexCoord.object target (matters because Object output uses its
    inverse matrix; a None target makes Object equivalent to Generated
    for world shading — i.e. the world-space ray direction).
  - Mapping.vector_type (Point / Texture / Vector / Normal)
  - Image colorspace and pixel reach
  - The actual fac socket source on MixShader (default vs linked)
"""
import bpy
import os

scene = bpy.context.scene
world = scene.world
nt = world.node_tree
print(f"=== world '{world.name}' ===")

mix = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeMixShader"), None)
print(f"\n-- MixShader '{mix.name}' --")
fac = mix.inputs[0]
print(f"  Fac is_linked={fac.is_linked} default={fac.default_value:.4f}")
if fac.is_linked:
    print(f"  Fac driver: {fac.links[0].from_node.bl_idname}.{fac.links[0].from_socket.name}")

for idx in (1, 2):
    sh = mix.inputs[idx]
    bg = sh.links[0].from_node if sh.is_linked else None
    if bg is None:
        print(f"  Shader[{idx}] not linked")
        continue
    print(f"\n-- {bg.bl_idname} '{bg.name}' --")
    s = bg.inputs.get("Strength")
    print(f"  Strength is_linked={s.is_linked} default={s.default_value:.4f}")
    color = bg.inputs.get("Color")
    src = color.links[0].from_node if color and color.is_linked else None
    if src is None:
        print(f"  Color unlinked")
        continue
    print(f"  Color from: {src.bl_idname} '{src.name}'")
    if src.bl_idname == "ShaderNodeTexImage":
        img = src.image
        print(f"    image='{img.name if img else None}' size={tuple(img.size) if img else None}")
        print(f"    colorspace='{img.colorspace_settings.name if img else None}'")
        print(f"    filepath='{img.filepath if img else None}'")
        print(f"    interpolation='{src.interpolation}' projection='{src.projection}' extension='{src.extension}'")
        # Walk vector socket
        vsock = src.inputs.get("Vector")
        if vsock and vsock.is_linked:
            mp = vsock.links[0].from_node
            print(f"    Vector source: {mp.bl_idname} '{mp.name}'")
            if mp.bl_idname == "ShaderNodeMapping":
                print(f"      vector_type='{mp.vector_type}'")
                loc = tuple(round(x, 4) for x in mp.inputs["Location"].default_value)
                rot = tuple(round(x, 4) for x in mp.inputs["Rotation"].default_value)
                scl = tuple(round(x, 4) for x in mp.inputs["Scale"].default_value)
                print(f"      Location={loc} (linked={mp.inputs['Location'].is_linked})")
                print(f"      Rotation={rot} (linked={mp.inputs['Rotation'].is_linked})  rot deg = {tuple(round(__import__('math').degrees(v),2) for v in rot)}")
                print(f"      Scale={scl} (linked={mp.inputs['Scale'].is_linked})")
                # Walk back further
                vsock2 = mp.inputs.get("Vector")
                if vsock2 and vsock2.is_linked:
                    tc = vsock2.links[0].from_node
                    print(f"      Vector source: {tc.bl_idname} '{tc.name}'")
                    if tc.bl_idname == "ShaderNodeTexCoord":
                        print(f"        TexCoord.object = {tc.object}")
                        # Print which output socket feeds the mapping
                        which = vsock2.links[0].from_socket.name
                        print(f"        TexCoord output used: '{which}'")
    elif src.bl_idname == "ShaderNodeTexSky":
        print(f"    sky_type='{src.sky_type}' sun_disc={src.sun_disc}")
        print(f"    sun_size={src.sun_size:.5f} sun_intensity={src.sun_intensity:.4f}")
        print(f"    sun_elevation={src.sun_elevation:.4f} ({__import__('math').degrees(src.sun_elevation):.2f} deg)")
        print(f"    sun_rotation={src.sun_rotation:.4f} ({__import__('math').degrees(src.sun_rotation):.2f} deg)")
        for attr in ("altitude", "air_density", "dust_density", "ozone_density",
                     "ground_albedo", "turbidity"):
            if hasattr(src, attr):
                print(f"    {attr}={getattr(src, attr):.4f}")
            else:
                print(f"    {attr}=<unsupported by sky_type>")
