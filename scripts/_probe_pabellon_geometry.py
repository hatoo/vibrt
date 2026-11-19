"""List the geometry around the water plane and dump material/visibility flags."""
import bpy

scene = bpy.context.scene

water = scene.objects.get("water_plane_still")
if water is None:
    raise SystemExit("no water_plane_still")
print(f"water: {water.location}, dimensions: {water.dimensions}")

# Find objects in the same Z neighborhood (the pond floor / ground / etc)
zw = water.location.z
print(f"\nObjects with Z in [{zw - 5}, {zw + 5}]:")
for obj in scene.objects:
    if obj.type != "MESH":
        continue
    if abs(obj.location.z - zw) > 5:
        continue
    mats = [s.material.name if s.material else "?" for s in obj.material_slots]
    print(f"  {obj.name:35s} loc={tuple(round(c,2) for c in obj.location)} dim={tuple(round(d,2) for d in obj.dimensions)} mats={mats}")

# Look for "ground" / "floor" / "stone" / "concrete" materials
print(f"\nMaterials with floor/ground/stone keywords:")
for mat in bpy.data.materials:
    nl = mat.name.lower()
    if any(k in nl for k in ("ground", "floor", "concrete", "marble", "stone", "rock", "plaster")):
        print(f"  {mat.name}")
