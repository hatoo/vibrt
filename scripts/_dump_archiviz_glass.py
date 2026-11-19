"""Find materials in flat_archiviz that look like window glass (transmission > 0
or Glass BSDF in source), and dump how vibrt's exporter mapped them.

Run via:
    blender -b test_scenes/flat_archiviz/flat_archiviz/flat-archiviz.blend \
        --python scripts/_dump_archiviz_glass.py
"""
import bpy, json, os, sys

sys.path.insert(0, os.path.abspath("blender"))
import vibrt_blender.exporter as exporter  # noqa: E402


def find_glass_in_tree(node_tree, depth=0, seen=None):
    """Walk node tree looking for Glass / Refraction / Transmission > 0."""
    if seen is None:
        seen = set()
    if node_tree is None or id(node_tree) in seen:
        return []
    seen.add(id(node_tree))
    hits = []
    for n in node_tree.nodes:
        t = n.bl_idname
        if t in ("ShaderNodeBsdfGlass", "ShaderNodeBsdfRefraction",
                 "ShaderNodeBsdfTransparent"):
            hits.append((depth, t, n.name))
        if t == "ShaderNodeBsdfPrincipled":
            try:
                tx = n.inputs.get("Transmission Weight") or n.inputs.get("Transmission")
                v = float(tx.default_value) if tx else 0.0
                if v > 0.01:
                    hits.append((depth, "Principled(transmission)", f"{n.name} t={v:.2f}"))
            except Exception:
                pass
        if t == "ShaderNodeGroup" and getattr(n, "node_tree", None):
            hits.extend(find_glass_in_tree(n.node_tree, depth + 1, seen))
    return hits


print("=== materials with glass-like nodes ===")
glass_mats = []
for mat in bpy.data.materials:
    if mat.node_tree is None:
        continue
    hits = find_glass_in_tree(mat.node_tree)
    if hits:
        glass_mats.append(mat.name)
        print(f"\nMATERIAL: {mat.name!r}")
        for d, t, n in hits:
            print(f"  {'  ' * d}{t}: {n}")

# Now run the exporter and look at how those mats got mapped.
print("\n=== exporter output for glass-like materials ===")
depsgraph = bpy.context.evaluated_depsgraph_get()
scene_json_str, _, _ = exporter.export_scene_to_memory(depsgraph)
scene = json.loads(scene_json_str)
mats_out = scene.get("materials", [])
for m in mats_out:
    if m.get("name") in glass_mats:
        # only print interesting fields
        keep = {}
        for k in ("name", "transmission", "ior", "roughness", "metallic",
                  "base_color", "alpha_threshold", "transmission_tex",
                  "specular_ior_level", "pure_glossy", "pure_diffuse",
                  "translucent_weight", "coat_weight", "volume"):
            if k in m and m[k] not in (None, 0, 0.0, [0.0, 0.0, 0.0],
                                       [1.0, 1.0, 1.0]):
                keep[k] = m[k]
        keep["name"] = m.get("name")
        print(json.dumps(keep, indent=2))

# Find which objects use those materials and where they are.
print("\n=== objects using glass materials ===")
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    for slot in obj.material_slots:
        if slot.material and slot.material.name in glass_mats:
            print(f"  {obj.name!r}: material {slot.material.name!r}, "
                  f"loc=({obj.location.x:.2f}, {obj.location.y:.2f}, {obj.location.z:.2f}), "
                  f"dim=({obj.dimensions.x:.2f}, {obj.dimensions.y:.2f}, {obj.dimensions.z:.2f})")
