"""Inventory the installed Blender shader nodes without opening a scene."""
import bpy
import json
import sys
from pathlib import Path

material = bpy.data.materials.new("vibrt_node_inventory")
material.use_nodes = True
tree = material.node_tree
result = {}
for name in sorted(dir(bpy.types)):
    if not name.startswith("ShaderNode"):
        continue
    try:
        node = tree.nodes.new(name)
    except RuntimeError:
        continue
    entry = {"label": node.bl_label, "inputs": [], "outputs": [], "enums": {}}
    for direction in ("inputs", "outputs"):
        entry[direction] = [
            {"name": s.name, "id": s.identifier, "type": s.type, "enabled": s.enabled}
            for s in getattr(node, direction)
        ]
    for prop in node.bl_rna.properties:
        if prop.type == "ENUM" and not prop.identifier.startswith("bl_") and prop.identifier not in {
            "type", "color_tag", "warning_propagation", "texture_mapping", "color_mapping"
        }:
            entry["enums"][prop.identifier] = [x.identifier for x in prop.enum_items]
    result[name] = entry
    tree.nodes.remove(node)
output = Path(sys.argv[sys.argv.index("--") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"blender": bpy.app.version_string, "nodes": result}, indent=2))
print(f"Inventoried {len(result)} shader node types: {output}")
