"""Compile Blender value/color/vector sockets to the shared C++ shader VM.

Python reads RNA and connections only. Numerical node implementations live in
vibrt-nodes. Unsupported sockets fail explicitly; they never become mean colors.
"""
import math

MATH = "ADD SUBTRACT MULTIPLY DIVIDE SINE COSINE TANGENT ARCSINE ARCCOSINE ARCTANGENT POWER LOGARITHM MINIMUM MAXIMUM ROUND LESS_THAN GREATER_THAN MODULO ABSOLUTE ARCTAN2 FLOOR CEIL FRACT SQRT INVERSE_SQRT SIGN EXPONENT RADIANS DEGREES SINH COSH TANH TRUNC SNAP WRAP COMPARE MULTIPLY_ADD PINGPONG SMOOTH_MIN SMOOTH_MAX FLOORED_MODULO".split()
VECTOR_MATH = "ADD SUBTRACT MULTIPLY DIVIDE CROSS_PRODUCT PROJECT REFLECT DOT_PRODUCT DISTANCE LENGTH SCALE NORMALIZE SNAP FLOOR CEIL MODULO FRACTION ABSOLUTE MINIMUM MAXIMUM WRAP SINE COSINE TANGENT REFRACT FACEFORWARD MULTIPLY_ADD POWER SIGN ROUND".split()
BLENDS = "MIX ADD MULTIPLY SUBTRACT SCREEN DIVIDE DIFFERENCE DARKEN LIGHTEN OVERLAY DODGE BURN HUE SATURATION VALUE COLOR SOFT_LIGHT LINEAR_LIGHT EXCLUSION".split()
COLOR_MODES = ["RGB", "HSV", "HSL"]


class UnsupportedNode(ValueError):
    pass


def _enum(value, choices):
    try:
        return choices.index(value)
    except ValueError as exc:
        raise UnsupportedNode(f"Unsupported mode {value!r}") from exc


class Compiler:
    def __init__(self, writer=None, textures=None, group_stack=()):
        self.writer = writer
        self.textures = textures
        self.groups = tuple(group_stack)
        self.nodes = []
        self.memo = {}
        self.constants = {}
        self.visiting = set()
        self.attributes = set()

    def add(self, op, inputs=(), modes=(), params=(), **kwargs):
        if len(self.nodes) >= 256:
            raise UnsupportedNode("Shader graph exceeds the 256 node limit")
        i = len(self.nodes)
        self.nodes.append(dict(op=op, inputs=list(inputs),
                               modes=list(modes) + [0] * (4-len(modes)),
                               params=list(params) + [0.0] * (4-len(params)), **kwargs))
        return i

    def constant(self, value):
        if isinstance(value, (int, float)):
            rgba = (float(value),) * 3 + (1.0,)
        else:
            rgba = tuple(float(v) for v in value)
            rgba = (rgba + (0.0,) * max(0, 3-len(rgba)) + (1.0,))[:4]
        if rgba not in self.constants:
            self.constants[rgba] = self.add("constant", params=rgba)
        return self.constants[rgba]

    def context(self, index):
        key = ("context", index)
        if key not in self.memo:
            self.memo[key] = self.add("context", modes=[index])
        return self.memo[key]

    def input(self, socket, groups=None, default=None):
        groups = self.groups if groups is None else groups
        if socket is None:
            if default is None:
                raise UnsupportedNode("Missing node input")
            return self.constant(default)
        if not socket.is_linked:
            return self.constant(socket.default_value if default is None else default)
        source = socket.links[0].from_socket
        result = self.output(source, groups)
        if socket.type in {"VALUE", "INT", "BOOLEAN"}:
            if source.type in {"RGBA", "VECTOR"}:
                result = self.add("convert", [result], [0 if source.type == "RGBA" else 1])
        return result

    def output(self, socket, groups=None):
        groups = self.groups if groups is None else groups
        node = socket.node
        key = (tuple(g.as_pointer() for g in groups), node.as_pointer(), socket.identifier)
        if key in self.memo:
            return self.memo[key]
        if key in self.visiting:
            raise UnsupportedNode(f"Cyclic shader graph at {node.name}.{socket.name}")
        self.visiting.add(key)
        try:
            result = self._node(node, socket, groups)
        except UnsupportedNode as exc:
            raise UnsupportedNode(f"{node.name} ({node.bl_idname}).{socket.name}: {exc}") from exc
        finally:
            self.visiting.remove(key)
        self.memo[key] = result
        return result

    def _node(self, n, out, groups):
        kind = n.bl_idname
        name = out.identifier
        def s(key, default=None):
            sock = n.inputs[key] if isinstance(key, int) else n.inputs.get(key)
            return self.input(sock, groups, default)
        def ins(*keys):
            return [s(k) for k in keys]
        def optional(key, value):
            return s(key) if n.inputs.get(key) is not None else self.constant(value)
        def coord(default=1):
            sock = n.inputs.get("Vector")
            if default == 0 and not (sock and sock.is_linked):
                self.attributes.add("uv:")
            return self.input(sock, groups) if sock and sock.is_linked else self.context(default)
        def prop(key, choices):
            return _enum(getattr(n, key), choices)
        def mode(key, words):
            return prop(key, words.split())
        if n.mute:
            link = next((l for l in n.internal_links if l.to_socket == out), None)
            if link is None:
                raise UnsupportedNode("Muted node has no internal bypass")
            return self.input(link.from_socket, groups)
        if kind == "NodeReroute":
            return s(0)
        if kind == "ShaderNodeGroup":
            if n.node_tree is None:
                raise UnsupportedNode("Missing node group")
            output = next((x for x in n.node_tree.nodes if x.bl_idname == "NodeGroupOutput" and x.is_active_output), None)
            if output is None:
                raise UnsupportedNode("Missing active group output")
            inner = next((x for x in output.inputs if x.identifier == out.identifier), None)
            return self.input(inner, groups + (n,))
        if kind == "NodeGroupInput":
            if not groups:
                raise UnsupportedNode("Unbound group input")
            outer = next((x for x in groups[-1].inputs if x.identifier == out.identifier), None)
            return self.input(outer, groups[:-1])
        if kind in {"ShaderNodeRGB", "ShaderNodeValue"}:
            return self.constant(out.default_value)
        if kind == "ShaderNodeMath":
            return self.add("math", ins(0, 1, 2), [prop("operation", MATH), int(n.use_clamp)])
        if kind == "ShaderNodeVectorMath":
            return self.add("vector_math", ins(0, 1, 2, 3), [prop("operation", VECTOR_MATH), int(out.type == "VALUE")])
        if kind in {"ShaderNodeMix", "ShaderNodeMixRGB"}:
            if kind == "ShaderNodeMixRGB":
                args, data, clamp_factor, clamp = ins(0, 1, 2), 0, True, n.use_clamp
            elif n.data_type == "RGBA":
                args, data, clamp_factor, clamp = ins(0, 6, 7), 0, n.clamp_factor, n.clamp_result
            elif n.data_type == "FLOAT":
                args, data, clamp_factor, clamp = ins(0, 2, 3), 1, n.clamp_factor, False
            elif n.data_type == "VECTOR":
                varying = n.factor_mode == "NON_UNIFORM"
                args, data, clamp_factor, clamp = ins(1 if varying else 0, 4, 5), 2 if varying else 1, n.clamp_factor, False
            else:
                raise UnsupportedNode(f"Mix data type {n.data_type}")
            return self.add("mix", args, [prop("blend_type", BLENDS) if data == 0 else 0, data, int(clamp_factor), int(clamp)])
        if kind == "ShaderNodeInvert":
            return self.add("invert", ins("Color", "Fac"))
        if kind == "ShaderNodeHueSaturation":
            return self.add("hsv", [s("Color"), s("Hue"), s("Saturation"), s("Value"), s("Fac" if "Fac" in n.inputs else "Factor")])
        if kind == "ShaderNodeGamma":
            return self.add("gamma", ins("Color", "Gamma"))
        if kind == "ShaderNodeBrightContrast":
            return self.add("bright_contrast", ins("Color", "Bright", "Contrast"))
        if kind in {"ShaderNodeCombineXYZ", "ShaderNodeCombineColor"}:
            return self.add("combine", ins(0, 1, 2), [prop("mode", COLOR_MODES) if kind.endswith("Color") else 0])
        if kind in {"ShaderNodeSeparateXYZ", "ShaderNodeSeparateColor"}:
            index = list(n.outputs).index(out)
            return self.add("separate", ins(0), [prop("mode", COLOR_MODES) if kind.endswith("Color") else 0, index])
        if kind == "ShaderNodeRGBToBW":
            return self.add("convert", ins(0), [0])
        if kind == "ShaderNodeClamp":
            return self.add("clamp", ins(0, 1, 2), [int(n.clamp_type == "RANGE")])
        if kind == "ShaderNodeMapRange":
            base = 6 if n.data_type == "FLOAT_VECTOR" else 0
            return self.add("map_range", ins(*range(base, base+6)), [mode("interpolation_type", "LINEAR STEPPED SMOOTHSTEP SMOOTHERSTEP"), int(n.clamp), int(base > 0)])
        if kind == "ShaderNodeMapping":
            return self.add("mapping", [s("Vector"), optional("Location", (0,0,0)), s("Rotation"), s("Scale")], [mode("vector_type", "POINT TEXTURE VECTOR NORMAL")])
        if kind == "ShaderNodeVectorRotate":
            return self.add("vector_rotate", ins(0, 1, 2, 3, 4), [mode("rotation_type", "AXIS_ANGLE X_AXIS Y_AXIS Z_AXIS EULER_XYZ"), int(n.invert)])
        if kind == "ShaderNodeVectorTransform":
            return self.add("vector_transform", ins(0), [mode("vector_type", "VECTOR POINT NORMAL"), mode("convert_from", "WORLD OBJECT CAMERA"), mode("convert_to", "WORLD OBJECT CAMERA")])
        if kind == "ShaderNodeSqueeze":
            return self.add("squeeze", ins(0, 1, 2))
        if kind == "ShaderNodeRadialTiling":
            return self.add("radial_tiling", ins(0, 1, 2), [list(n.outputs).index(out), int(n.normalize)])
        if kind == "ShaderNodeCameraData":
            return self.add("camera", modes=[{"View Vector":0,"View Z Depth":1,"View Distance":2}[name]])
        if kind == "ShaderNodeLightPath":
            fields = {"Is Camera Ray":(20,0), "Is Shadow Ray":(20,1), "Is Diffuse Ray":(20,2), "Is Glossy Ray":(20,3), "Is Singular Ray":(21,0), "Is Reflection Ray":(21,1), "Is Transmission Ray":(21,2), "Ray Length":(18,0), "Ray Depth":(18,1), "Diffuse Depth":(18,2), "Glossy Depth":(18,3), "Transmission Depth":(19,0)}
            if name not in fields:
                raise UnsupportedNode(f"Path output {name} requires additional integrator state")
            context, channel = fields[name]
            return self.add("separate", [self.context(context)], [0, channel])
        if kind == "ShaderNodeLightFalloff":
            return self.add("light_falloff", ins(0, 1), [{"Quadratic":0,"Linear":1,"Constant":2}[name]])
        if kind == "ShaderNodeTexNoise":
            return self.add("noise", [coord(), optional("W", 0), s("Scale"), s("Detail"), s("Roughness"), s("Lacunarity"), optional("Offset", 0), optional("Gain", 1), s("Distortion")],
                            [int(n.noise_dimensions[0]), mode("noise_type", "MULTIFRACTAL FBM HYBRID_MULTIFRACTAL RIDGED_MULTIFRACTAL HETERO_TERRAIN"), int(n.normalize), int(name == "Color")])
        if kind == "ShaderNodeTexVoronoi":
            selector = {"Distance": 0, "Color": 1, "Position": 2, "W": 3, "Radius": 4}[name]
            args = [s(key) if n.inputs.get(key) else self.constant(value) for key, value in
                    (("W", 0), ("Scale", 5), ("Detail", 0), ("Roughness", 0.5), ("Lacunarity", 2), ("Smoothness", 1), ("Exponent", 0.5), ("Randomness", 1))]
            return self.add("voronoi", [coord()] + args,
                            [int(n.voronoi_dimensions[0]), mode("feature", "F1 F2 SMOOTH_F1 DISTANCE_TO_EDGE N_SPHERE_RADIUS"), mode("distance", "EUCLIDEAN MANHATTAN CHEBYCHEV MINKOWSKI"), selector + 5*int(n.normalize)])
        if kind == "ShaderNodeTexWave":
            return self.add("wave", [coord()] + ins("Scale", "Distortion", "Detail", "Detail Scale", "Detail Roughness", "Phase Offset"),
                            [mode("wave_type", "BANDS RINGS"), mode("bands_direction", "X Y Z DIAGONAL"), mode("rings_direction", "X Y Z SPHERICAL"), mode("wave_profile", "SIN SAW TRI")])
        if kind == "ShaderNodeTexChecker":
            return self.add("checker", [coord()] + ins("Color1", "Color2", "Scale"), [int(name == "Fac")])
        if kind == "ShaderNodeTexBrick":
            return self.add("brick", [coord()] + ins(*range(1, 10)), [int(name == "Fac")],
                            [n.offset, n.offset_frequency, n.squash, n.squash_frequency])
        if kind == "ShaderNodeTexGradient":
            return self.add("gradient", [coord()], [mode("gradient_type", "LINEAR QUADRATIC EASING DIAGONAL RADIAL QUADRATIC_SPHERE SPHERICAL")])
        if kind == "ShaderNodeTexMagic":
            return self.add("magic", [coord()] + ins("Scale", "Distortion"), [n.turbulence_depth, int(name == "Fac")])
        if kind == "ShaderNodeTexWhiteNoise":
            return self.add("white_noise", [optional("Vector", (0,0,0)), optional("W", 0)], [int(n.noise_dimensions[0]), int(name == "Color")])
        if kind == "ShaderNodeTexGabor":
            return self.add("gabor", [coord()] + ins(1, 2, 3, 4, 5), [mode("gabor_type", "2D 3D"), {"Value": 0, "Phase": 1, "Intensity": 2}[name]])
        if kind in {"ShaderNodeTexImage", "ShaderNodeTexEnvironment"}:
            if n.image is None:
                return self.constant(1.0 if name == "Alpha" else (1.0, 0.0, 1.0, 1.0))
            if n.image.source not in {"FILE", "GENERATED", "SEQUENCE", "MOVIE"}:
                raise UnsupportedNode(f"Image source {n.image.source} needs a texture manager")
            from .material_export import export_image_texture
            tex = export_image_texture(n.image, self.writer, self.textures)
            environment = kind.endswith("Environment")
            associate = n.image.alpha_mode not in {"NONE", "CHANNEL_PACKED"} and not n.image.colorspace_settings.is_data
            return self.add("image", [coord(3 if environment else 0)],
                            [mode("interpolation", "Linear Closest Cubic Smart"), 0 if environment else mode("extension", "REPEAT EXTEND CLIP MIRROR"), 0 if environment else mode("projection", "FLAT BOX SPHERE TUBE"), int(name == "Alpha")],
                            [getattr(n, "projection_blend", 0.0), (3.0 if n.projection == "MIRROR_BALL" else 1.0) if environment else 0.0,
                             float(associate and n.image.alpha_mode == "STRAIGHT"),
                             float(associate and bool(n.outputs.get("Alpha") and n.outputs["Alpha"].is_linked))], texture=tex)
        if kind == "ShaderNodeTexSky":
            from . import vibrt_native
            from .material_export import _PreBakedTexture, _export_prebaked
            settings = {key: getattr(n, key) for key in (
                "turbidity", "ground_albedo", "sun_elevation", "sun_rotation", "sun_size", "sun_intensity", "altitude", "air_density", "aerosol_density", "ozone_density", "sun_disc")}
            settings.update(model=n.sky_type, sun_direction=tuple(n.sun_direction))
            key = "shader-sky:" + repr(settings)
            tex = next((i for i,t in enumerate(self.textures) if t.get("_key") == key), None)
            if tex is None:
                pixels = vibrt_native.sky_pixels(settings, 2048, 1024)
                tex = _export_prebaked(_PreBakedTexture(pixels, 2048, 1024, key), self.writer, self.textures)
            return self.add("image", [coord(1)], [0, 0, 0, 0], [0, 2], texture=tex)
        if kind in {"ShaderNodeRGBCurve", "ShaderNodeFloatCurve", "ShaderNodeVectorCurve"}:
            cm = n.mapping
            cm.update()
            lo, hi = cm.clip_min_x, cm.clip_max_x
            if hi <= lo:
                raise UnsupportedNode("Invalid curve domain")
            table = []
            for i in range(257):
                x = lo + (hi-lo)*i/256
                if kind == "ShaderNodeRGBCurve":
                    combined = cm.evaluate(cm.curves[3], x)
                    table.extend([cm.evaluate(cm.curves[ch], combined) for ch in range(3)] + [1.0])
                elif kind == "ShaderNodeVectorCurve":
                    table.extend([cm.evaluate(cm.curves[ch], x) for ch in range(3)] + [1.0])
                else:
                    value = cm.evaluate(cm.curves[0], x)
                    table.extend([value]*3 + [1.0])
            return self.add("curve", [s(1), s(0)], [0, int(cm.extend == "EXTRAPOLATED")], [lo, hi], table=table)
        if kind == "ShaderNodeValToRGB":
            ramp = n.color_ramp
            table = [float(c) for i in range(257) for c in ramp.evaluate(i/256)]
            return self.add("ramp", ins(0), [int(ramp.interpolation == "CONSTANT"), int(name == "Alpha")], table=table)
        if kind in {"ShaderNodeBlackbody", "ShaderNodeWavelength"}:
            return self.add("blackbody" if kind.endswith("Blackbody") else "wavelength", ins(0))
        if kind == "ShaderNodeTexCoord":
            if n.object is not None or n.from_instancer:
                raise UnsupportedNode("Explicit coordinate object / instancer coordinates")
            contexts = {"UV": 0, "Generated": 1, "Object": 2, "Normal": 13, "Camera": 12, "Window": 14, "Reflection": 15}
            if name == "UV":
                self.attributes.add("uv:")
            return self.context(contexts[name])
        if kind == "ShaderNodeUVMap":
            if n.from_instancer:
                raise UnsupportedNode("Instancer UV")
            self.attributes.add("uv:" + n.uv_map)
            return self.context(0)
        if kind in {"ShaderNodeVertexColor", "ShaderNodeAttribute"}:
            attr = n.layer_name if kind.endswith("VertexColor") else n.attribute_name
            if kind.endswith("Attribute") and n.attribute_type != "GEOMETRY":
                raise UnsupportedNode("Only mesh geometry attributes are available")
            self.attributes.add("color:" + attr)
            index = self.context(8)
            return self.add("convert", [index], [2 if name == "Alpha" else 1]) if name in {"Alpha", "Fac"} else index
        if kind == "ShaderNodeNewGeometry":
            contexts = {"Position": 3, "Normal": 4, "True Normal": 5, "Incoming": 6, "Tangent": 7, "Backfacing": 16}
            if name not in contexts:
                raise UnsupportedNode(f"Geometry output {name} needs extra mesh data")
            return self.context(contexts[name])
        if kind == "ShaderNodeObjectInfo":
            if name == "Material Index":
                import bpy
                from .material_export import _CURRENT_MATERIAL
                material = bpy.data.materials.get(_CURRENT_MATERIAL)
                return self.constant(material.pass_index if material else 0)
            if name == "Location":
                return self.context(10)
            if name == "Color":
                return self.context(11)
            return self.add("separate", [self.context(9)], [0, {"Random": 0, "Object Index": 1, "Material Index": 2, "Alpha": 3}[name]])
        if kind == "ShaderNodeFresnel":
            normal = s("Normal") if n.inputs["Normal"].is_linked else self.context(4)
            return self.add("fresnel", [s("IOR"), normal])
        if kind == "ShaderNodeLayerWeight":
            normal = s("Normal") if n.inputs["Normal"].is_linked else self.context(4)
            return self.add("layer_weight", [s("Blend"), normal], [int(name == "Facing")])
        if kind == "ShaderNodeNormalMap":
            if n.space == "TANGENT":
                self.attributes.add("uv:" + n.uv_map)
            return self.add("normal_map", ins("Color", "Strength"),
                            [mode("space", "TANGENT OBJECT WORLD BLENDER_OBJECT BLENDER_WORLD"),
                             int(getattr(n, "convention", "OPENGL") == "DIRECTX")])
        if kind == "ShaderNodeNormal":
            return self.add("normal", [s(0) if n.inputs[0].is_linked else self.context(4)], [int(name == "Dot")], list(n.outputs["Normal"].default_value))
        if kind == "ShaderNodeTangent":
            if n.direction_type == "UV_MAP":
                self.attributes.add("uv:" + n.uv_map)
            return self.add("tangent", modes=[int(n.direction_type == "RADIAL"), mode("axis", "X Y Z")])
        raise UnsupportedNode("No evaluator for this node/output")

    def graph(self, outputs):
        return {"nodes": self.nodes, "outputs": outputs}

    def attach_attributes(self, params):
        colors = sorted(a[6:] for a in self.attributes if a.startswith("color:"))
        uvs = sorted(a[3:] for a in self.attributes if a.startswith("uv:"))
        if len(colors) > 1 or len(uvs) > 1:
            raise UnsupportedNode("Multiple color/UV layers require additional mesh attributes")
        if colors:
            params["_vertex_color_attr"] = colors[0]
        if uvs:
            params["_uv_map"] = uvs[0]


PRINCIPLED_INPUTS = {
    "Base Color": "base_color", "Roughness": "roughness", "Metallic": "metallic",
    "IOR": "ior", "Transmission Weight": "transmission", "Emission Color": "emission",
    "Emission Strength": "emission_strength", "Alpha": "alpha", "Specular IOR Level": "specular_ior_level",
    "Anisotropic": "anisotropy", "Anisotropic Rotation": "tangent_rotation",
    "Coat Weight": "coat_weight", "Coat Roughness": "coat_roughness", "Coat IOR": "coat_ior",
    "Sheen Weight": "sheen_weight", "Sheen Roughness": "sheen_roughness", "Sheen Tint": "sheen_tint",
    "Subsurface Weight": "sss_weight", "Subsurface Radius": "sss_radius",
    "Normal": "normal", "Tangent": "tangent",
}


def attach(node, params, writer, textures, group_stack=()):
    """Attach all connected value sockets, preserving existing closure behavior."""
    kind = node.bl_idname
    mapping = dict(PRINCIPLED_INPUTS) if kind == "ShaderNodeBsdfPrincipled" else {
        "Color": "emission" if kind == "ShaderNodeEmission" else "base_color",
        "Strength": "emission_strength", "Roughness": "roughness", "IOR": "ior",
        "Normal": "normal", "Anisotropy": "anisotropy", "Rotation": "tangent_rotation",
        "Tangent": "tangent", "Offset": "hair_offset", "RoughnessU": "hair_roughness_u", "RoughnessV": "hair_roughness_v",
    }
    compiler = Compiler(writer, textures, group_stack)
    if kind == "ShaderNodeBsdfSheen":
        mapping.update(Color="sheen_tint", Roughness="sheen_roughness")
    elif kind == "ShaderNodeBsdfMetallic":
        mapping["Base Color"] = "base_color"
    outputs = {}
    for name, target in mapping.items():
        sock = node.inputs.get(name)
        if sock is None or not sock.enabled or not sock.is_linked:
            continue
        # The caller reports precise unsupported sockets rather than guessing a value.
        source = sock.links[0].from_node
        if target == "normal" and source.bl_idname == "ShaderNodeBump":
            outputs["bump_height"] = compiler.input(source.inputs["Height"])
            outputs["bump_strength"] = compiler.input(source.inputs["Strength"])
            distance = compiler.input(source.inputs["Distance"])
            if source.invert:
                distance = compiler.add("math", [distance, compiler.constant(-1), compiler.constant(0)], [MATH.index("MULTIPLY")])
            outputs["bump_distance"] = distance
            if source.inputs["Normal"].is_linked:
                outputs["normal"] = compiler.input(source.inputs["Normal"])
        else:
            outputs[target] = compiler.input(sock)
    if not outputs:
        return params
    # Emission needs its separate strength even when only one input is linked.
    if "emission" in outputs or "emission_strength" in outputs:
        color = node.inputs.get("Emission Color") if kind == "ShaderNodeBsdfPrincipled" else node.inputs.get("Color")
        strength = node.inputs.get("Emission Strength") if kind == "ShaderNodeBsdfPrincipled" else node.inputs.get("Strength")
        outputs["emission"] = compiler.input(color)
        outputs["emission_strength"] = compiler.input(strength, default=1.0 if strength is None else None)
    compiler.attach_attributes(params)
    params["shader_graph"] = compiler.graph(outputs)
    if "base_color" in outputs:
        params.pop("color_graph", None)
        params.pop("base_color_tex", None)
        params["use_vertex_color"] = False
    if "alpha" in outputs:
        params["alpha_threshold"] = 0.0
    if "normal" in outputs or "bump_height" in outputs:
        params.pop("normal_tex", None)
        params.pop("bump_tex", None)
    return params


def compile_leaf(node, writer, textures, group_stack=()):
    """Fast path: compile sockets before the legacy image-baking traversal."""
    kind = node.bl_idname
    if kind == "ShaderNodeBsdfMetallic" and node.fresnel_type != "F82":
        raise UnsupportedNode("Metallic physical-conductor IOR/extinction requires a conductor closure")
    if kind not in {"ShaderNodeBsdfPrincipled", "ShaderNodeBsdfDiffuse", "ShaderNodeBsdfAnisotropic", "ShaderNodeBsdfGlossy", "ShaderNodeBsdfGlass", "ShaderNodeBsdfRefraction", "ShaderNodeBsdfTranslucent", "ShaderNodeBsdfTransparent", "ShaderNodeEmission", "ShaderNodeBsdfMetallic"}:
        return None
    params = {"base_color": [0.8]*3, "metallic": 0.0, "roughness": 0.5, "ior": 1.45, "transmission": 0.0, "emission": [0.0]*3}
    mapping = PRINCIPLED_INPUTS if kind == "ShaderNodeBsdfPrincipled" else {"Color": "emission" if kind == "ShaderNodeEmission" else "base_color", "Roughness": "roughness", "IOR": "ior", "Anisotropy": "anisotropy", "Rotation": "tangent_rotation"}
    for name, field in mapping.items():
        socket = node.inputs.get(name)
        if socket is None or not socket.enabled or field in {"normal", "tangent", "emission_strength"}:
            continue
        value = socket.default_value
        value = [float(v) for v in value[:3]] if socket.type in {"RGBA", "VECTOR"} else float(value)
        if field == "tangent_rotation":
            value *= 2*math.pi
        params["alpha_blend" if field == "alpha" else field] = value
    if kind == "ShaderNodeBsdfMetallic":
        params["base_color"] = list(node.inputs["Base Color"].default_value[:3])
    if kind in {"ShaderNodeBsdfGlossy", "ShaderNodeBsdfAnisotropic", "ShaderNodeBsdfMetallic"}:
        params.update(metallic=1.0, pure_glossy=True)
    elif kind in {"ShaderNodeBsdfDiffuse", "ShaderNodeBsdfTranslucent"}:
        params.update(pure_diffuse=True, specular_ior_level=0.0)
        if kind == "ShaderNodeBsdfTranslucent":
            params["translucent_weight"] = 1.0
    elif kind in {"ShaderNodeBsdfGlass", "ShaderNodeBsdfRefraction"}:
        params["transmission"] = 1.0
    elif kind == "ShaderNodeBsdfTransparent":
        params["alpha_blend"] = 0.0
    elif kind == "ShaderNodeEmission":
        params.update(base_color=[0.0]*3, pure_diffuse=True, specular_ior_level=0.0)
    strength = node.inputs.get("Emission Strength" if kind == "ShaderNodeBsdfPrincipled" else "Strength")
    if strength is not None:
        params["emission"] = [c*float(strength.default_value) for c in params["emission"]]
    return attach(node, params, writer, textures, group_stack)
