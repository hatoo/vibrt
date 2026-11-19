"""For each pixel of the pabellon bake, find the equivalent world
direction and check what the bake stores there. Specifically check
camera-forward and reflected-camera direction to see if the sunset
orange is in the right place."""
import bpy, os, sys, math
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender.exporter as exporter

scene = bpy.context.scene
world = scene.world
exporter._SKY_BAKE_CACHE.clear()
exporter.prebake_sky_envmaps_for_world(world)

key = exporter._world_generic_cache_key(world) if exporter._world_needs_full_bake(world) else exporter._sky_node_cache_key(world, exporter._world_sky_node(world))
cached = exporter._SKY_BAKE_CACHE[key]
rgb, w, h, sun = cached
print(f"bake {w}x{h}")

# Kernel reads bake at:
#   theta = v * pi   (v from buffer-y/h, where buffer y=0 is bottom-up = nadir
#                     in our convention because of the bake camera rotation)
#   phi = u * 2pi    (u from buffer-x/w)
def kernel_sample(theta_rad, phi_rad):
    u = (phi_rad / (2 * math.pi)) % 1.0
    v = theta_rad / math.pi
    x = min(int(u * w), w - 1)
    y = min(int(v * h), h - 1)
    return rgb[y, x, :]

# Camera forward direction in world space
cam = scene.camera
fwd = -cam.matrix_world.col[2].xyz
fx, fy, fz = fwd.x, fwd.y, fwd.z
phi_fwd = math.atan2(fy, fx)
theta_fwd = math.acos(max(-1, min(1, fz)))
print(f"\ncamera fwd = ({fx:.3f}, {fy:.3f}, {fz:.3f})")
print(f"  spherical: theta={math.degrees(theta_fwd):.1f}deg phi={math.degrees(phi_fwd):.1f}deg")
print(f"  bake at fwd: rgb={tuple(round(float(c),3) for c in kernel_sample(theta_fwd, phi_fwd))}")

# Reflected ray from foreground water (camera ray hits water plane, reflects)
# Camera fwd has fz≈0 (horizontal), so a ray going slightly down (e.g. from
# bottom of frame) has fz<0. Reflect: flip z.
# Test angle 20° below horizon.
ang_below = math.radians(20)
horiz_len = math.cos(ang_below)
ray_down = (fx * horiz_len, fy * horiz_len, -math.sin(ang_below))
# Normalize and reflect (flip z)
refl = (ray_down[0], ray_down[1], -ray_down[2])
refl_norm = math.sqrt(sum(c*c for c in refl))
refl = tuple(c/refl_norm for c in refl)
phi_r = math.atan2(refl[1], refl[0])
theta_r = math.acos(max(-1, min(1, refl[2])))
print(f"\nreflected (foreground) = {tuple(round(c,3) for c in refl)}")
print(f"  spherical: theta={math.degrees(theta_r):.1f}deg phi={math.degrees(phi_r):.1f}deg")
print(f"  bake at refl: rgb={tuple(round(float(c),3) for c in kernel_sample(theta_r, phi_r))}")

# Cardinal directions
for name, (dx, dy, dz) in [("+X", (1,0,0)), ("+Y", (0,1,0)), ("-X", (-1,0,0)),
                            ("-Y", (0,-1,0)), ("+Z (zenith)", (0,0,1)),
                            ("-Z (nadir)", (0,0,-1))]:
    ph = math.atan2(dy, dx)
    th = math.acos(max(-1, min(1, dz)))
    c = kernel_sample(th, ph)
    print(f"  {name:14s}: rgb={tuple(round(float(x),3) for x in c)}")
