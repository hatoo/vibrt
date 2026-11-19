"""Dump lone_monk's compositor (Blender 5.x has scene.compositing_node_group)."""
import bpy

scene = bpy.context.scene

# Try multiple paths
nt = None
for attr in ("compositing_node_group", "node_tree"):
    if hasattr(scene, attr):
        candidate = getattr(scene, attr)
        if candidate is not None:
            nt = candidate
            print(f"compositor via scene.{attr}: {nt!r}")
            break

if nt is None:
    print("no compositor node tree found")
    # Maybe it's hidden in another property
    for attr in dir(scene):
        if "node" in attr.lower() or "compos" in attr.lower():
            try:
                v = getattr(scene, attr)
                if v is not None and hasattr(v, "nodes"):
                    print(f"  scene.{attr} = {v!r}")
            except Exception:
                pass

if nt is not None:
    print(f"\n--- nodes ({len(nt.nodes)}) ---")
    for n in nt.nodes:
        print(f"  {n.bl_idname}/{n.name}")
        for inp in n.inputs:
            try:
                if inp.is_linked:
                    src = inp.links[0].from_node
                    print(f"    in {inp.name} ← {src.bl_idname}/{src.name}")
                else:
                    v = inp.default_value
                    if hasattr(v, "__iter__"):
                        v = tuple(round(float(x), 4) for x in v)
                    else:
                        v = round(float(v), 4) if isinstance(v, (int, float)) else v
                    print(f"    in {inp.name} = {v}")
            except Exception as ex:
                print(f"    in {inp.name} (err: {ex})")
        # Dump some special properties
        if n.bl_idname == "CompositorNodeColorBalance":
            for attr in ("lift", "gamma", "gain", "offset", "power", "slope"):
                if hasattr(n, attr):
                    v = getattr(n, attr)
                    if hasattr(v, "__iter__"):
                        v = tuple(round(float(x), 4) for x in v)
                    print(f"    .{attr} = {v}")
        if n.bl_idname == "CompositorNodeMath":
            print(f"    operation = {n.operation}")
        if n.bl_idname == "CompositorNodeMixRGB":
            print(f"    blend_type = {n.blend_type}")
