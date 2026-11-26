"""Evaluate supported sky graphs in-process, without creating Blender datablocks.

Graph support is checked before evaluation. Unsupported graphs keep the isolated
Cycles worker; set VIBRT_CYCLES_SKY=1 to force that reference path for comparisons.
"""

import os
import time

import numpy as np

from . import node_ops
from .export_context import ExportContext


class UnsupportedSky(Exception):
    pass


def _scalar(socket, depth=0):
    if socket is None or depth > 24:
        raise UnsupportedSky("missing or recursive scalar input")
    if not socket.is_linked:
        return float(socket.default_value)
    node = socket.links[0].from_node
    if node.mute:
        raise UnsupportedSky("muted scalar node")
    if node.bl_idname == "ShaderNodeValue":
        return float(node.outputs[0].default_value)
    if node.bl_idname == "NodeReroute":
        return _scalar(node.inputs[0], depth + 1)
    if node.bl_idname == "ShaderNodeMath":
        a, b = (_scalar(s, depth + 1) for s in node.inputs[:2])
        if node.operation in {"ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "MINIMUM", "MAXIMUM"}:
            return float(node_ops.apply(dict(type="math", op=node.operation.lower(), b=b,
                                            clamp=node.use_clamp), [a, a, a])[0])
    raise UnsupportedSky(f"scalar input from {node.bl_idname}")


def _rgb_curve(node):
    """Cycles' 257-entry composite/channel table, including HDR extrapolation."""
    mapping = node.mapping
    mapping.update()
    low = min(curve.points[0].location.x for curve in mapping.curves)
    high = max(curve.points[-1].location.x for curve in mapping.curves)
    if high <= low:
        raise UnsupportedSky("empty RGB curve domain")
    values = np.linspace(low, high, 257, dtype=np.float32)
    table = np.array([
        [mapping.evaluate(channel, mapping.evaluate(mapping.curves[3], float(v)))
         for v in values] for channel in mapping.curves[:3]
    ], dtype=np.float32)
    extrapolate = mapping.extend == "EXTRAPOLATED"
    factor = _scalar(node.inputs[0])

    settings = dict(type="rgb_curve", lut=table.ravel().tolist(), low=low, high=high,
                    fac=factor, extrapolate=extrapolate)
    return lambda rgb: node_ops.apply(settings, rgb)


class _Graph:
    def __init__(self, native, *, context=None):
        self.export_context = context if context is not None else ExportContext()
        self.native = native
        self.skies = []
        self.pixels = {}

    def sky(self, node):
        if node.mute:
            raise UnsupportedSky("muted Sky Texture")
        mapping = node.texture_mapping
        if (any(mapping.translation) or any(mapping.rotation)
                or tuple(mapping.scale) != (1.0, 1.0, 1.0)
                or (mapping.mapping_x, mapping.mapping_y, mapping.mapping_z) != ("X", "Y", "Z")
                or mapping.use_min or mapping.use_max):
            raise UnsupportedSky("Sky Texture coordinate mapping")
        vector = node.inputs.get("Vector")
        invert = False
        if vector and vector.is_linked:
            link = vector.links[0]
            if (link.from_node.bl_idname != "ShaderNodeTexCoord"
                    or link.from_socket.name not in {"Generated", "Reflection"}
                    or link.from_node.object is not None or link.from_node.mute):
                raise UnsupportedSky("Sky Texture vector input")
            # In a world shader Texture Coordinate.Reflection is -ray.direction.
            invert = link.from_socket.name == "Reflection"
        if node.sky_type not in {"PREETHAM", "HOSEK_WILKIE", "SINGLE_SCATTERING", "MULTIPLE_SCATTERING"}:
            raise UnsupportedSky(f"sky model {node.sky_type}")
        settings = {name: getattr(node, name) for name in (
            "turbidity", "ground_albedo", "sun_elevation", "sun_rotation",
            "sun_size", "sun_intensity", "altitude", "air_density",
            "aerosol_density", "ozone_density", "sun_disc",
        )}
        settings.update(model=node.sky_type, sun_direction=tuple(node.sun_direction), invert_direction=invert)
        self.skies.append(node)
        identity = node.as_pointer()

        def evaluate(w, h):
            key = identity, w, h
            if key not in self.pixels:
                self.pixels[key] = self.native.sky_pixels(settings, w, h)
            return self.pixels[key]
        return evaluate

    def color(self, socket, depth=0):
        from . import material_export

        if socket is None or depth > 24:
            raise UnsupportedSky("missing or recursive color input")
        if not socket.is_linked:
            value = np.array(socket.default_value[:3], dtype=np.float32)
            return lambda w, h: value
        node = socket.links[0].from_node
        if node.mute:
            raise UnsupportedSky(f"muted {node.bl_idname}")
        kind = node.bl_idname
        if kind == "ShaderNodeTexSky":
            return self.sky(node)
        if kind == "ShaderNodeRGB":
            value = np.array(node.outputs[0].default_value[:3], dtype=np.float32)
            return lambda w, h: value
        if kind == "NodeReroute":
            return self.color(node.inputs[0], depth + 1)
        if kind in {"ShaderNodeMixRGB", "ShaderNodeMix"}:
            factor, a, b, blend, clamp, ok = material_export._mix_node_params(node)
            if not ok or blend not in material_export._SUPPORTED_MIX_BLENDS:
                raise UnsupportedSky("unsupported color mix")
            fac = _scalar(factor)
            if kind == "ShaderNodeMixRGB" or node.clamp_factor:
                fac = min(max(fac, 0.0), 1.0)
            left, right = self.color(a, depth + 1), self.color(b, depth + 1)
            return lambda w, h: material_export._apply_mix_blend(left(w, h), right(w, h), fac, blend, clamp)
        if kind == "ShaderNodeRGBCurve":
            source = self.color(node.inputs["Color"], depth + 1)
            transform = _rgb_curve(node)
            return lambda w, h: transform(source(w, h))
        if kind == "ShaderNodeHueSaturation":
            source = self.color(node.inputs["Color"], depth + 1)
            hue, saturation, value, factor = (_scalar(node.inputs[i]) for i in range(4))

            settings = dict(type="hue_sat", hue=hue, saturation=saturation, value=value,
                            fac=factor, clamp_negative=True)
            return lambda w, h: node_ops.apply(settings, source(w, h))
        if kind == "ShaderNodeGamma":
            source = self.color(node.inputs["Color"], depth + 1)
            gamma = _scalar(node.inputs["Gamma"])

            return lambda w, h: node_ops.apply(dict(type="gamma", gamma=gamma), source(w, h))
        if kind == "ShaderNodeInvert":
            source = self.color(node.inputs["Color"], depth + 1)
            fac = _scalar(node.inputs[0])
            return lambda w, h: node_ops.apply(dict(type="invert", fac=fac), source(w, h))
        raise UnsupportedSky(f"color node {kind}")

    def shader(self, node, depth=0):
        if node is None or node.mute or depth > 24:
            raise UnsupportedSky("missing, muted or recursive world shader")
        kind = node.bl_idname
        if kind in {"ShaderNodeBackground", "ShaderNodeEmission"}:
            color = self.color(node.inputs["Color"], depth + 1)
            strength = _scalar(node.inputs["Strength"])
            return lambda w, h: node_ops.apply(dict(type="math", op="multiply", b=strength), color(w, h))
        if kind in {"ShaderNodeMixShader", "ShaderNodeAddShader"}:
            inputs = [s for s in node.inputs if s.type == "SHADER"]
            if len(inputs) != 2 or any(not s.is_linked for s in inputs):
                raise UnsupportedSky("unconnected world mix")
            a, b = [self.shader(s.links[0].from_node, depth + 1) for s in inputs]
            if kind == "ShaderNodeAddShader":
                return lambda w, h: node_ops.apply(dict(type="mix", blend="add", fac=1.0), a(w, h), b(w, h))
            factor = min(max(_scalar(node.inputs[0]), 0.0), 1.0)
            return lambda w, h: node_ops.apply(dict(type="mix", fac=factor), a(w, h), b(w, h))
        raise UnsupportedSky(f"world shader {kind}")

    def shader_with_sun(self, node):
        first = len(self.skies)
        evaluate = self.shader(node)
        skies = {sky.as_pointer(): sky for sky in self.skies[first:]}
        return evaluate, next(iter(skies.values())) if len(skies) == 1 else None


def prepare_scene(scene, *, cancel, log, width=None, height=None, context):
    """Return True only if every required environment map can be evaluated."""
    import bpy
    from . import exporter, runner, sky_bake

    if os.environ.get("VIBRT_CYCLES_SKY") == "1":
        return False
    native = runner.find_native_module()
    if native is None or not hasattr(native, "sky_pixels"):
        return False
    # The coefficient tables and color conversion are verified against this
    # Blender version and its default working space; other versions/configs bake.
    if (bpy.app.version[:2] != (5, 1) or os.environ.get("OCIO")
            or bpy.data.colorspace.working_space_interop_id != "lin_rec709_scene"):
        return False
    world = scene.world
    graph = _Graph(native, context=context)
    plans = []
    w, h = (2048, 1024) if width is None else (width, height)
    try:
        if sky_bake._needs_world_bake(world):
            direct = exporter._world_sky_node(world)
            split = exporter._detect_mix_world_camera_ray_split(world)
            mixed = exporter._detect_mix_world(world)
            if direct is not None:
                # Background.Strength is applied once by _export_world.
                plans.append((exporter._sky_node_cache_key(world, direct), graph.sky(direct), direct, w, h))
            elif split is not None:
                for branch, role in zip(split, ("lighting", "camera")):
                    plans.append((exporter._world_branch_cache_key(world, branch, role),
                                  graph.shader(branch), None, w, h))
            elif mixed is not None and exporter._mixed_world_uses_layers(mixed):
                for branch in mixed[:2]:
                    sky = branch.inputs["Color"].links[0].from_node
                    if sky.bl_idname == "ShaderNodeTexSky":
                        plans.append((exporter._sky_node_alone_cache_key(world, sky), graph.sky(sky), sky, w, h))
            else:
                output = next((n for n in world.node_tree.nodes
                               if n.bl_idname == "ShaderNodeOutputWorld" and n.is_active_output), None)
                if output is None or not output.inputs["Surface"].is_linked:
                    raise UnsupportedSky("missing world output")
                evaluate, sun = graph.shader_with_sun(output.inputs["Surface"].links[0].from_node)
                plans.append((exporter._world_generic_cache_key(world), evaluate, sun, w, h))
        portals = exporter._portal_emissions_to_bake(scene)
        if portals:
            mat, emission = portals[0]
            evaluate, sun = graph.shader_with_sun(emission)
            pw, ph = (1024, 512) if width is None else (width, height)
            plans.append((exporter._portal_emission_cache_key(scene.name, mat.name, emission.name),
                          evaluate, sun, pw, ph))
    except UnsupportedSky as ex:
        log(f"vibrt: sky graph uses {ex}; preparing it with Cycles")
        return False

    cache = {}
    start = time.perf_counter()
    for key, evaluate, sky, w, h in plans:
        if cancel():
            raise KeyboardInterrupt("Sky preparation cancelled")
        rgb = np.broadcast_to(evaluate(w, h), (h, w, 3)).astype(np.float32, order="C", copy=True)
        if not np.isfinite(rgb).all():
            raise RuntimeError("Native sky produced non-finite radiance")
        sun = exporter._extract_sun_from_bake_inplace(rgb, w, h, sky, key) if sky is not None else None
        cache[key] = rgb, w, h, sun
    if cancel():
        raise KeyboardInterrupt("Sky preparation cancelled")
    context.sky_maps.update(cache)
    if plans:
        models = ", ".join(sorted({n.sky_type for n in graph.skies}))
        log(f"[vibrt] native sky {models}: {len(plans)} environment map(s) in {time.perf_counter()-start:.3f}s")
    return True
