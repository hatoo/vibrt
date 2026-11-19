"""Iterate the depsgraph and dump xy positions of every pebbles instance
to see whether the 20k particles are correctly spread across the pool
floor or clumped in one area."""
import bpy
import numpy as np

scene = bpy.context.scene
scene.frame_set(scene.frame_current)
deps = bpy.context.evaluated_depsgraph_get()

xs, ys, zs = [], [], []
for inst in deps.object_instances:
    if not inst.is_instance:
        continue
    parent = inst.parent
    if parent is None or "pebbles_scatter" not in parent.name:
        continue
    m = inst.matrix_world
    xs.append(m[0][3])
    ys.append(m[1][3])
    zs.append(m[2][3])

xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)
print(f"pebble instances: {len(xs)}")
print(f"  x range: [{xs.min():.2f}, {xs.max():.2f}], mean {xs.mean():.2f}, std {xs.std():.2f}")
print(f"  y range: [{ys.min():.2f}, {ys.max():.2f}], mean {ys.mean():.2f}, std {ys.std():.2f}")
print(f"  z range: [{zs.min():.2f}, {zs.max():.2f}], mean {zs.mean():.2f}, std {zs.std():.2f}")
# Histogram in x and y to see clumping
print("  x quartile counts:")
qs = np.percentile(xs, [25, 50, 75])
print(f"    quartiles at x = {qs[0]:.2f}, {qs[1]:.2f}, {qs[2]:.2f}")
# Bin to 10 columns to see distribution
bins_x = np.linspace(xs.min(), xs.max(), 11)
hist_x, _ = np.histogram(xs, bins=bins_x)
print(f"  x bins (10): {hist_x.tolist()}")
bins_y = np.linspace(ys.min(), ys.max(), 11)
hist_y, _ = np.histogram(ys, bins=bins_y)
print(f"  y bins (10): {hist_y.tolist()}")
