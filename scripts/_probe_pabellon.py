"""List worlds and scenes in the loaded .blend, and dump each world's
shader-graph topology so we can pick a vibrt-friendly one."""
import bpy

print(f"=== scenes ({len(bpy.data.scenes)}) ===")
for sc in bpy.data.scenes:
    print(f"  {sc.name}  active_world={sc.world.name if sc.world else '(none)'}")

print(f"\n=== worlds ({len(bpy.data.worlds)}) ===")
for w in bpy.data.worlds:
    print(f"\n[{w.name}]")
    if not w.use_nodes or w.node_tree is None:
        print("  (no nodes)")
        continue
    out = next((n for n in w.node_tree.nodes
                if n.bl_idname == "ShaderNodeOutputWorld"), None)
    if out is None or not out.inputs["Surface"].is_linked:
        print("  (no World Output / unlinked)")
        continue
    src = out.inputs["Surface"].links[0].from_node
    print(f"  surface ← {src.bl_idname} '{src.name}'")
    # Walk back one or two levels for context
    for sock in src.inputs:
        if sock.is_linked:
            ln = sock.links[0]
            print(f"    {sock.name:12s} <- {ln.from_node.bl_idname} '{ln.from_node.name}'")
        else:
            try:
                dv = sock.default_value
                if hasattr(dv, '__len__'):
                    val = "(" + ", ".join(f"{v:.3f}" for v in dv) + ")"
                else:
                    val = f"{dv:.4f}" if isinstance(dv, float) else str(dv)
                print(f"    {sock.name:12s} = {val}")
            except Exception:
                pass
    # Walk one more level into linked inputs.
    for sock in src.inputs:
        if not sock.is_linked:
            continue
        deeper = sock.links[0].from_node
        print(f"  {deeper.bl_idname} '{deeper.name}':")
        for inner in deeper.inputs:
            if inner.is_linked:
                ln = inner.links[0]
                print(f"    {inner.name:12s} <- {ln.from_node.bl_idname} '{ln.from_node.name}'")
            else:
                try:
                    dv = inner.default_value
                    if hasattr(dv, '__len__'):
                        val = "(" + ", ".join(f"{v:.3f}" for v in dv) + ")"
                    else:
                        val = f"{dv:.4f}" if isinstance(dv, float) else str(dv)
                    print(f"    {inner.name:12s} = {val}")
                except Exception:
                    pass
