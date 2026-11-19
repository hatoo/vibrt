"""Show ALL collections / scene tree, including hidden / excluded layers."""
import bpy

scene = bpy.context.scene
print(f"=== scene: {scene.name!r} ===")

def walk_collection(coll, depth=0):
    ind = "  " * depth
    print(f"{ind}{coll.name!r} (hide_viewport={coll.hide_viewport}, hide_render={coll.hide_render})")
    for obj in coll.objects:
        h = ""
        if obj.hide_render:
            h = " HIDDEN_RENDER"
        if obj.hide_viewport:
            h = " HIDDEN_VIEW"
        print(f"{ind}  - {obj.name!r} ({obj.type}){h}")
    for child in coll.children:
        walk_collection(child, depth + 1)

walk_collection(scene.collection)

# Check view layers
print(f"\n=== view layers ===")
for vl in scene.view_layers:
    print(f"  {vl.name!r}")
    for layer_coll in vl.layer_collection.children:
        print(f"    {layer_coll.name!r}: exclude={layer_coll.exclude}, hide={layer_coll.hide_viewport}")
