"""Dump lone_monk's full scene-level configuration: compositor,
view transform, color management, render passes — anything that could
silently scale render output."""
import bpy

scene = bpy.context.scene
print(f"=== scene {scene.name!r} ===")
print(f"  resolution: {scene.render.resolution_x} × {scene.render.resolution_y} @ {scene.render.resolution_percentage}%")

print(f"\n=== view settings ===")
vs = scene.view_settings
print(f"  view_transform = {vs.view_transform}")
print(f"  look = {vs.look}")
print(f"  exposure = {vs.exposure}")
print(f"  gamma = {vs.gamma}")

print(f"\n=== display settings ===")
ds = scene.display_settings
print(f"  display_device = {ds.display_device}")

print(f"\n=== sequencer color ===")
try:
    print(f"  sequencer_colorspace_settings.name = {scene.sequencer_colorspace_settings.name}")
except Exception:
    pass

print(f"\n=== compositor ===")
print(f"  use_nodes (compositor) = {scene.use_nodes}")
if scene.node_tree is not None:
    nt = scene.node_tree
    print(f"  node_tree.nodes count = {len(nt.nodes)}")
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
            except Exception:
                pass

print(f"\n=== cycles settings ===")
cy = scene.cycles
for attr in ("samples", "use_adaptive_sampling", "diffuse_bounces", "glossy_bounces",
             "transmission_bounces", "volume_bounces", "transparent_max_bounces",
             "max_bounces", "blur_glossy", "sample_clamp_direct", "sample_clamp_indirect",
             "use_light_tree", "use_camera_cull", "use_distance_cull", "use_animated_seed",
             "seed", "film_exposure", "ao_bounces", "ao_factor"):
    if hasattr(cy, attr):
        print(f"  {attr} = {getattr(cy, attr)}")

print(f"\n=== render settings ===")
print(f"  film_transparent = {scene.render.film_transparent}")
print(f"  pixel_aspect_x/y = {scene.render.pixel_aspect_x} / {scene.render.pixel_aspect_y}")

# View Layer / Pass settings
print(f"\n=== view layers ===")
for vl in scene.view_layers:
    print(f"  layer {vl.name!r}")
    print(f"    use_pass_combined = {vl.use_pass_combined}")
    if hasattr(vl, "samples"):
        print(f"    samples = {vl.samples}")
