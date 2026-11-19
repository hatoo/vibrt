"""Count depsgraph instances by source object name to see whether the lotus,
pebble, and tree particle systems actually populate the iterator."""
import bpy
from collections import Counter

bpy.context.scene.frame_set(bpy.context.scene.frame_current)
deps = bpy.context.evaluated_depsgraph_get()

instance_count: Counter = Counter()
direct_count: Counter = Counter()
total = 0
for inst in deps.object_instances:
    total += 1
    if inst.is_instance:
        src = inst.parent
        instance_count[src.name if src else "(none)"] += 1
    else:
        direct_count[inst.object.name] += 1

print(f"total iterator yields: {total}")
print(f"direct (non-instance) objects: {sum(direct_count.values())}")
print(f"instances: {sum(instance_count.values())}")
print(f"\n=== top-10 instance parents ===")
for name, n in instance_count.most_common(10):
    print(f"  {name:40s}: {n}")
print(f"\n=== top-15 direct objects ===")
for name, n in direct_count.most_common(15):
    print(f"  {name:40s}: {n}")
