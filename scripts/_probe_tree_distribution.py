"""Where are tree-particle instances in world space? Compare to the
camera frustum so we can see whether trees on the camera-right side of
the pool actually exist in the depsgraph."""
import bpy
import bpy_extras
import mathutils

scene = bpy.context.scene
deps = bpy.context.evaluated_depsgraph_get()
cam = scene.camera

xs, ys, zs = [], [], []
in_screen = 0
behind = 0
for di in deps.object_instances:
    if not di.is_instance:
        continue
    if di.parent is None or di.parent.name != "tree_scatter_plane":
        continue
    p = di.matrix_world.translation
    xs.append(p.x); ys.append(p.y); zs.append(p.z)
    co = bpy_extras.object_utils.world_to_camera_view(scene, cam, p)
    if co.z > 0 and 0 <= co.x <= 1 and 0 <= co.y <= 1:
        in_screen += 1
    if co.z <= 0:
        behind += 1

import statistics
print(f"tree instances: {len(xs)}, in-screen={in_screen}, behind-cam={behind}")
print(f"  x: min={min(xs):.2f} max={max(xs):.2f}")
print(f"  y: min={min(ys):.2f} max={max(ys):.2f}")
print(f"  z: min={min(zs):.2f} max={max(zs):.2f}")
print(f"camera at x={cam.matrix_world.translation.x:.2f} y={cam.matrix_world.translation.y:.2f}")

# Histogram x in 8 bins
import collections
h = collections.Counter()
xmin, xmax = min(xs), max(xs)
binw = (xmax - xmin) / 8
for x in xs:
    idx = min(int((x - xmin) / binw), 7)
    h[idx] += 1
print(f"  x histogram (8 bins from {xmin:.2f} to {xmax:.2f}):")
for i in range(8):
    print(f"    bin{i}: {h[i]}")
