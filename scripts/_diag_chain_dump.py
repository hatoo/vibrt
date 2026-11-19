"""Headless diagnostic: dump the exported Base-Color color_graph for
lone_monk materials by driving `_try_emit_color_graph` directly on named
bpy materials.

Answers: does vibrt emit each chain as a per-pixel color_graph (with a
terminal HueSaturation Value applied), or return None and fall back to a
baked-texture path that drops the Value?

Run:  blender -b lone_monk.blend --python scripts/_diag_chain_dump.py [-- MatName ...]
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "blender"))
for _m in list(sys.modules):
    if _m == "vibrt_blender" or _m.startswith("vibrt_blender."):
        del sys.modules[_m]

import bpy  # noqa: E402
from vibrt_blender import material_export as ME  # noqa: E402
from vibrt_blender.bin_writer import BinWriter  # noqa: E402

argv = sys.argv
targets = argv[argv.index("--") + 1:] if "--" in argv else None


def principled_basecolor_sock(mat):
    nt = mat.node_tree
    if nt is None:
        return None
    for n in nt.nodes:
        if n.bl_idname == "ShaderNodeBsdfPrincipled":
            return n.inputs.get("Base Color")
    return None


def chain_desc(sock, depth=0, seen=None):
    """Compact textual description of the Blender node chain feeding sock."""
    if seen is None:
        seen = set()
    sock = ME._follow_reroutes(sock)
    if not sock.is_linked:
        try:
            v = list(sock.default_value)[:3]
        except Exception:
            v = sock.default_value
        return f"const{v}"
    link = sock.links[0]
    src = link.from_node
    bl = src.bl_idname.replace("ShaderNode", "")
    key = src.as_pointer()
    tag = f"{bl}<{link.from_socket.name}>"
    if bl == "HueSaturation":
        h = src.inputs["Hue"].default_value
        s = src.inputs["Saturation"].default_value
        v = src.inputs["Value"].default_value
        f = src.inputs["Fac"].default_value
        tag = f"HueSat(H={h:.3f},S={s:.3f},V={v:.3f},F={f:.3f})"
    if key in seen or depth > 8:
        return tag + "{...}"
    seen.add(key)
    parts = []
    for inp in src.inputs:
        if inp.is_linked:
            parts.append(f"{inp.name}={chain_desc(inp, depth + 1, seen)}")
    inner = (" [" + ", ".join(parts) + "]") if parts else ""
    return tag + inner


names = targets or [m.name for m in bpy.data.materials]
for nm in names:
    mat = bpy.data.materials.get(nm)
    if mat is None:
        print(f"\n### {nm!r}: NOT FOUND")
        continue
    bc = principled_basecolor_sock(mat)
    if bc is None:
        print(f"\n### {nm!r}: no Principled Base Color socket")
        continue
    print(f"\n### {nm!r}")
    print("  chain:", chain_desc(bc))
    writer = BinWriter()
    textures = []
    try:
        graph = ME._try_emit_color_graph(bc, writer, textures) if bc.is_linked else None
    except Exception as e:
        print(f"  _try_emit_color_graph RAISED: {type(e).__name__}: {e}")
        continue
    if graph is None:
        print("  -> graph=None  (FALLS BACK to baked-texture path; HueSat Value at risk)")
    else:
        nodes = graph.get("nodes", graph) if isinstance(graph, dict) else graph
        out = graph.get("output") if isinstance(graph, dict) else None
        print(f"  -> graph: {len(nodes)} nodes, output={out}")
        for i, nd in enumerate(nodes):
            extra = {k: v for k, v in nd.items() if k not in ("type", "lut")}
            if "lut" in nd:
                extra["lut_len"] = len(nd["lut"])
            print(f"     [{i}] {nd.get('type')}  {extra}")

print("\n=== DONE ===", flush=True)
