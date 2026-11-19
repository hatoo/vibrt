"""Scan a Blender scene for materials with constant Alpha < 1 that previously
broke the renderer (metallic=1 + alpha<1 → mirror window bug). Prints any
matches plus the resolved exporter output so we can verify alpha_blend now
catches them all.

Run from a per-scene Makefile rule, e.g.:
    blender -b test_scenes/<scene>/<scene>/<scene>.blend \
        --python scripts/_scan_alpha_metallic_bugs.py
"""
import bpy, json, os, sys

sys.path.insert(0, os.path.abspath("blender"))
import vibrt_blender.exporter as exporter  # noqa: E402

# Find Blender materials with constant Alpha < 1 in their Principled BSDF.
print(f"=== {bpy.context.scene.name}: alpha<1 Principled materials ===")
hits = []
for mat in bpy.data.materials:
    if mat.node_tree is None:
        continue
    for n in mat.node_tree.nodes:
        if n.bl_idname != "ShaderNodeBsdfPrincipled":
            continue
        a = n.inputs.get("Alpha")
        if a is None or a.is_linked:
            continue
        av = float(a.default_value)
        if av < 0.999:
            m_in = n.inputs.get("Metallic")
            t_in = n.inputs.get("Transmission Weight") or n.inputs.get("Transmission")
            mv = float(m_in.default_value) if m_in and not m_in.is_linked else None
            tv = float(t_in.default_value) if t_in and not t_in.is_linked else None
            hits.append((mat.name, av, mv, tv))
            print(f"  {mat.name!r}: alpha={av:.3f} metallic={mv} transmission={tv}")

if not hits:
    print("  (none)")
    raise SystemExit

# Now check that exporter sets alpha_blend correctly.
print(f"\n=== exported alpha_blend values ===")
depsgraph = bpy.context.evaluated_depsgraph_get()
scene_json_str, _, _ = exporter.export_scene_to_memory(depsgraph)
scene = json.loads(scene_json_str)
mats_out = scene["materials"]

# Build name → index by walking depsgraph (same order as exporter)
mat_first_seen = {}
ord_idx = 0
for obj_inst in depsgraph.object_instances:
    obj = obj_inst.object
    if obj is None or obj.type != "MESH":
        continue
    for slot in obj.material_slots:
        if slot.material and slot.material.name not in mat_first_seen:
            mat_first_seen[slot.material.name] = ord_idx
            ord_idx += 1

for name, av, mv, tv in hits:
    if name in mat_first_seen:
        idx = mat_first_seen[name]
        if idx < len(mats_out):
            m = mats_out[idx]
            ab = m.get("alpha_blend", "(missing)")
            at = m.get("alpha_threshold", 0.0)
            print(f"  {name!r}: alpha_blend={ab} alpha_threshold={at}")
        else:
            print(f"  {name!r}: idx={idx} OUT OF RANGE")
    else:
        print(f"  {name!r}: not used by any mesh in depsgraph")
