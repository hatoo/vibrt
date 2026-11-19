"""Count depsgraph instances by particle parent so we can confirm
trees / lotus / pebbles are all spread out the way they should be in
a Cycles render."""
import bpy
from collections import Counter

deps = bpy.context.evaluated_depsgraph_get()

# Original-name counter, separated by whether is_instance
direct = Counter()
inst_by_parent = Counter()

for di in deps.object_instances:
    if di.is_instance:
        # Particle system instance
        pname = di.parent.name if di.parent else "<no parent>"
        inst_by_parent[pname] += 1
    else:
        direct[di.object.type] += 1

print("=== direct objects in deps (non-instance) ===")
for k, v in direct.most_common():
    print(f"  {k}: {v}")

print("\n=== particle-instance counts by parent ===")
for k, v in inst_by_parent.most_common():
    print(f"  {k}: {v}")
