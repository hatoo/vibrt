"""Find which object the camera's left-edge corner ray hits ~2m out, in the
flat-archiviz scene. We use Blender's ray_cast against the evaluated depsgraph
to identify the mesh + material vibrt is shading as black."""
import bpy
from mathutils import Vector

scene = bpy.context.scene
deps = bpy.context.evaluated_depsgraph_get()
cam = scene.camera
cam_mat = cam.matrix_world
eye = cam_mat.translation

right = cam_mat.col[0].to_3d().normalized()
up    = cam_mat.col[1].to_3d().normalized()
fwd   = (-cam_mat.col[2].to_3d()).normalized()

cd = cam.data
# Blender's view_frame for the proper aspect.
W, H = scene.render.resolution_x, scene.render.resolution_y
shift_x = cd.shift_x
shift_y = cd.shift_y

# Replicate exporter._export_camera math.
frame = cd.view_frame(scene=scene)
norm = [(p.x / -p.z, p.y / -p.z) for p in frame]
xs = [n[0] for n in norm]; ys = [n[1] for n in norm]
half_w = (max(xs) - min(xs)) * 0.5
half_h = (max(ys) - min(ys)) * 0.5
cx = (max(xs) + min(xs)) * 0.5
cy = (max(ys) + min(ys)) * 0.5
shift_x_ndc = cx / (2.0 * half_w) if half_w > 0 else 0
shift_y_ndc = cy / (2.0 * half_h) if half_h > 0 else 0
import math
fov_y = 2.0 * math.atan(half_h)
print(f"image {W}x{H}  half_h={half_h:.4f}  half_w={half_w:.4f}  shift_ndc=({shift_x_ndc:.4f},{shift_y_ndc:.4f})")

# Build a ray for each diagnostic pixel.  In the kernel, idx.x=0/idx.y=0 is
# the BOTTOM-LEFT pixel of the kernel buffer (which becomes the bottom of the
# Blender RenderResult). PIL-y=0 is the TOP of the saved PNG, so PIL_y maps
# to kernel idx.y = (H-1) - PIL_y.
def kernel_dir(idx_x, idx_y):
    px = (2.0 * (idx_x + 0.5) / W) - 1.0 + 2.0 * shift_x_ndc
    py = (2.0 * (idx_y + 0.5) / H) - 1.0 + 2.0 * shift_y_ndc
    aspect = W / H
    half_w_dev = math.tan(fov_y * 0.5) * aspect
    half_h_dev = math.tan(fov_y * 0.5)
    U = right * half_w_dev
    V = up * half_h_dev
    d = U * px + V * py + fwd
    return d.normalized()

probes = [
    ("TL_corner",  0,    0),       # PIL y=0  → kernel idx.y = H-1
    ("TL_arc",     0,    150),
    ("ML_edge",    H//2, 0),
    ("BL_corner",  H-1,  0),       # PIL y=H-1 → kernel idx.y = 0
]

for label, pil_y, pil_x in probes:
    kix = pil_x
    kiy = (H - 1) - pil_y     # kernel y is flipped relative to PIL
    d = kernel_dir(kix, kiy)
    # Raycast the depsgraph from eye in direction d, max range 100m.
    res, hit_loc, hit_normal, hit_index, hit_obj, hit_mat = scene.ray_cast(
        deps, eye, d, distance=100.0,
    )
    print(f"\n--- {label} (PIL y={pil_y} x={pil_x}) ---")
    print(f"  dir         = ({d.x:+.4f}, {d.y:+.4f}, {d.z:+.4f})")
    if not res:
        print(f"  ray_cast: NO HIT within 100m")
        continue
    t = (hit_loc - eye).length
    print(f"  hit obj     = {hit_obj.name}")
    print(f"  hit dist    = {t:.4f} m")
    print(f"  hit world   = ({hit_loc.x:.3f}, {hit_loc.y:.3f}, {hit_loc.z:.3f})")
    print(f"  hit normal  = ({hit_normal.x:+.3f}, {hit_normal.y:+.3f}, {hit_normal.z:+.3f})")
    # Material at hit
    if hit_index >= 0 and hit_obj.data is not None:
        # Material slot index for this triangle.
        try:
            poly = hit_obj.data.polygons[hit_index]
            slot = poly.material_index
            mat = hit_obj.material_slots[slot].material if slot < len(hit_obj.material_slots) else None
            print(f"  material    = {mat.name if mat else '(none)'}")
        except Exception as ex:
            print(f"  material    = (?: {ex})")
    print(f"  obj.visible_camera = {getattr(hit_obj, 'visible_camera', '?')}")
    print(f"  obj.hide_viewport = {hit_obj.hide_viewport}")
    print(f"  obj.hide_render   = {hit_obj.hide_render}")
