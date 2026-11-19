"""Find which mesh objects use each material to localise where 'church
brick' actually shows up in the rendered image."""
import bpy

scene = bpy.context.scene

# For each material, find using objects + their bbox
materials_of_interest = (
    "church brick", "brick clay", "brick marble", "brick marble smooth",
    "brick pavement", "column marble", "plaster med old", "plaster old",
    "plaster no brick", "stone pavement", "outdoor pavement", "roof",
    "terracotta",
)

for mat_name in materials_of_interest:
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        continue
    print(f"\n=== {mat_name} ===")
    for obj in scene.objects:
        if obj.type != "MESH" or not obj.material_slots:
            continue
        for slot in obj.material_slots:
            if slot.material and slot.material.name == mat_name:
                # bbox in world space
                import mathutils
                vs = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
                minc = (min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs))
                maxc = (max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs))
                size = (maxc[0]-minc[0], maxc[1]-minc[1], maxc[2]-minc[2])
                print(f"  obj={obj.name}: bbox center=({(minc[0]+maxc[0])/2:.2f}, "
                      f"{(minc[1]+maxc[1])/2:.2f}, {(minc[2]+maxc[2])/2:.2f}) "
                      f"size=({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}) "
                      f"vis_camera={obj.visible_camera}")
                break

# Camera info
cam = scene.camera
m = cam.matrix_world
fwd = (-m.col[2][0], -m.col[2][1], -m.col[2][2])
print(f"\n=== camera ===")
print(f"  loc = ({cam.location.x:.2f}, {cam.location.y:.2f}, {cam.location.z:.2f})")
print(f"  forward = ({fwd[0]:.3f}, {fwd[1]:.3f}, {fwd[2]:.3f})")
