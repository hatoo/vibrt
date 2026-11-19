"""Print z-coordinates of water plane vs pebbles vs pond floor so we can
see if the pebbles are actually below the water and visible to a
refracted camera ray."""
import bpy

def show(obj):
    bb = [obj.matrix_world @ v.co if hasattr(v, 'co') else None for v in []]
    # Use evaluated bounding box
    bbox_world = [obj.matrix_world @ corner_local for corner_local in
                  [type(obj.location)(c) for c in obj.bound_box]]
    zs = [v.z for v in bbox_world]
    print(f"  {obj.name:40s} loc={tuple(round(x,3) for x in obj.location)} "
          f"z_min={min(zs):.3f} z_max={max(zs):.3f} dim={tuple(round(d,3) for d in obj.dimensions)}")

print("=== top-level pool-related objects ===")
for name in ("water_plane_still", "lotus_scattering_plane", "pebbles_scatter",
             "tree_scatter_plane"):
    obj = bpy.data.objects.get(name)
    if obj:
        show(obj)

print("\n=== first 5 pebble instances (collection 'pebbles') ===")
col = bpy.data.collections.get("pebbles")
if col:
    for i, obj in enumerate(col.objects[:5]):
        show(obj)
        if i >= 4:
            break

# Where does the camera-to-water ray refract to?
import math
cam = bpy.context.scene.camera
cam_loc = cam.location
print(f"\ncamera at {tuple(round(x,3) for x in cam_loc)}")
forward_world = -cam.matrix_world.col[2].xyz
print(f"camera forward = {tuple(round(x,3) for x in forward_world)}")
water_z = 1.293
# Find where camera_ray intersects water_z plane
t = (water_z - cam_loc.z) / forward_world.z if forward_world.z != 0 else None
if t and t > 0:
    hit = (cam_loc.x + t * forward_world.x, cam_loc.y + t * forward_world.y, water_z)
    print(f"camera ray hits water at t={t:.3f}, point={tuple(round(x,3) for x in hit)}")
    # IOR=1.1 refraction angle
    dot = forward_world.dot(type(forward_world)((0, 0, -1)))
    cos_i = abs(dot)
    sin_i = math.sqrt(max(0, 1-cos_i*cos_i))
    sin_t = sin_i / 1.1  # n1=1.0 (air), n2=1.1 (water)
    cos_t = math.sqrt(1 - sin_t*sin_t)
    print(f"  incidence cos={cos_i:.3f} angle={math.degrees(math.acos(cos_i)):.1f}°")
    print(f"  refract  cos={cos_t:.3f} angle={math.degrees(math.acos(cos_t)):.1f}°")
