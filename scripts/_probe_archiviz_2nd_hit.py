"""For the TL black-corner ray, get the SECOND surface hit (the one past the
front face of Wall Basement Floor) — that's what vibrt is seeing once t_min
skips the first 1.7m hit."""
import bpy, math
from mathutils import Vector

scene = bpy.context.scene
deps = bpy.context.evaluated_depsgraph_get()
cam = scene.camera
eye = cam.matrix_world.translation
right = cam.matrix_world.col[0].to_3d().normalized()
up = cam.matrix_world.col[1].to_3d().normalized()
fwd = (-cam.matrix_world.col[2].to_3d()).normalized()

cd = cam.data
W, H = scene.render.resolution_x, scene.render.resolution_y
frame = cd.view_frame(scene=scene)
norm = [(p.x / -p.z, p.y / -p.z) for p in frame]
xs = [n[0] for n in norm]; ys = [n[1] for n in norm]
half_w = (max(xs) - min(xs)) * 0.5
half_h = (max(ys) - min(ys)) * 0.5
cx = (max(xs)+min(xs))*0.5; cy = (max(ys)+min(ys))*0.5
shift_x_ndc = cx / (2.0 * half_w) if half_w > 0 else 0
shift_y_ndc = cy / (2.0 * half_h) if half_h > 0 else 0
fov_y = 2.0 * math.atan(half_h)
aspect = W / H
half_w_dev = math.tan(fov_y * 0.5) * aspect
half_h_dev = math.tan(fov_y * 0.5)
U = right * half_w_dev
V = up * half_h_dev

def kernel_dir(idx_x, idx_y):
    px = (2.0 * (idx_x + 0.5) / W) - 1.0 + 2.0 * shift_x_ndc
    py = (2.0 * (idx_y + 0.5) / H) - 1.0 + 2.0 * shift_y_ndc
    return (U * px + V * py + fwd).normalized()

def material_for_hit(obj, poly_index):
    """Resolve material by trying both polygons and loop_triangles indices."""
    me = obj.data
    if me is None:
        return None
    # Try polygon index first
    try:
        if 0 <= poly_index < len(me.polygons):
            slot = me.polygons[poly_index].material_index
            return obj.material_slots[slot].material if slot < len(obj.material_slots) else None
    except Exception:
        pass
    # Try loop_triangles (depsgraph-eval geom usually delivers tri-indices)
    try:
        me.calc_loop_triangles()
        if 0 <= poly_index < len(me.loop_triangles):
            tri = me.loop_triangles[poly_index]
            slot = me.polygons[tri.polygon_index].material_index
            return obj.material_slots[slot].material if slot < len(obj.material_slots) else None
    except Exception as e:
        return f"(err {e})"
    return None

# Cast multiple successive rays, advancing past each hit, to map all faces
# the camera ray passes through up to 5m.
def trace_through(eye, d, max_t=5.0):
    origin = Vector(eye)
    eps = 1e-4
    total = 0.0
    hops = []
    while total < max_t and len(hops) < 8:
        res, hit_loc, hit_n, hit_idx, hit_obj, _ = scene.ray_cast(deps, origin, d, distance=max_t-total)
        if not res:
            break
        seg = (hit_loc - origin).length
        total += seg
        # Resolve material - need to use the EVALUATED object
        eval_obj = hit_obj.evaluated_get(deps) if not hit_obj.is_evaluated else hit_obj
        mat = material_for_hit(eval_obj, hit_idx)
        hops.append((total, hit_obj.name, mat, hit_n.copy(), hit_loc.copy()))
        origin = hit_loc + d * eps
    return hops

probes = [
    ("TL_corner",  0,    0),
    ("TL_arc",     0,    150),
    ("ML_edge",    H//2, 0),
    ("BL_corner",  H-1,  0),
]

for label, pil_y, pil_x in probes:
    kix = pil_x
    kiy = (H - 1) - pil_y
    d = kernel_dir(kix, kiy)
    print(f"\n=== {label} (PIL y={pil_y} x={pil_x}) dir=({d.x:+.3f},{d.y:+.3f},{d.z:+.3f}) ===")
    hops = trace_through(eye, d, max_t=6.0)
    for i, (t, name, mat, n, loc) in enumerate(hops):
        front = n.dot(d) < 0
        face = "FRONT" if front else "BACK "
        mn = mat.name if mat is not None and not isinstance(mat, str) else (str(mat) if mat else "(?)")
        print(f"  hop[{i}] t={t:6.3f}m  {face}  obj={name:35s}  mat={mn:25s}  n=({n.x:+.2f},{n.y:+.2f},{n.z:+.2f})  loc=({loc.x:+.2f},{loc.y:+.2f},{loc.z:+.2f})")
