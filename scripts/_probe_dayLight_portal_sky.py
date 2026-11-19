"""Dump dayLight_portal's TexSky controls + scene sun direction so we
can tell whether the original Sun light and the TexSky's sun disc
overlap (= double-counted)."""
import bpy
import math

mat = bpy.data.materials["dayLight_portal"]
sky = next((n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeTexSky"), None)
if sky is None:
    print("no TexSky in dayLight_portal")
else:
    print(f"=== dayLight_portal TexSky ===")
    for attr in ("sky_type", "sun_disc", "sun_size", "sun_intensity",
                 "sun_elevation", "sun_rotation", "altitude",
                 "air_density", "aerosol_density", "ozone_density",
                 "ground_albedo", "turbidity"):
        if hasattr(sky, attr):
            v = getattr(sky, attr)
            print(f"  {attr} = {v}")

# TexSky sun direction (Cycles convention):
#   x = cos(elev) * sin(rot)
#   y = cos(elev) * cos(rot)
#   z = sin(elev)
if sky is not None:
    elev = sky.sun_elevation
    rot = sky.sun_rotation
    sun_x = math.cos(elev) * math.sin(rot)
    sun_y = math.cos(elev) * math.cos(rot)
    sun_z = math.sin(elev)
    print(f"\n  TexSky sun direction (toward sun): ({sun_x:.4f}, {sun_y:.4f}, {sun_z:.4f})")

# Scene sun light
print(f"\n=== Scene Sun lights ===")
for obj in bpy.context.scene.objects:
    if obj.type == "LIGHT" and obj.data.type == "SUN":
        m = obj.matrix_world
        # local -Z = direction the sun light shines toward (away from "sun")
        z = m.col[2]
        forward = (-z[0], -z[1], -z[2])  # photons travel along this
        toward_sun = (z[0], z[1], z[2])  # surface → sun
        print(f"  {obj.name}: strength={obj.data.energy:.3f} angle={obj.data.angle:.4f}")
        print(f"    photon direction = ({forward[0]:.4f}, {forward[1]:.4f}, {forward[2]:.4f})")
        print(f"    toward-sun       = ({toward_sun[0]:.4f}, {toward_sun[1]:.4f}, {toward_sun[2]:.4f})")
