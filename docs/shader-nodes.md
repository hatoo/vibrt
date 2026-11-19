# Shader node support

Target: Blender 5.1.1 / Cycles v5.1.1. This table inventories all 99 installed
`ShaderNode` types. **This is not complete Cycles compatibility.** Native means
the node executes in the new graph; partial and legacy entries retain the
specific limitations below. Unsupported value graphs produce an exporter
warning and may fall back to the older material conversion. Unsupported BSDFs
also warn. Check the console rather than assuming a fallback is equivalent.

## Architecture and limits

Python traverses connections and reads images/curve tables. `vibrt-nodes`
validates and packs the graph; one C++ implementation runs on CPU and CUDA.
Each graph supports 256 value instructions and multiple material outputs;
cycles, bad references, invalid modes and oversized graphs are rejected.
Mix/Add support up to 16 closure leaves and 8 levels. GPU stack sizes are
queried from OptiX. Only used operations enter the per-scene compiled module.
PTX is cached in the system temporary directory and OptiX maintains its own
compiled-module cache. New operation/closure combinations still incur a cold
compile: the tested Classroom scene reached its first preview in about 29 s
with compilation and 1.4 s on the next render (RTX 5080, 25%, 4 samples).

Math implements 41 operations; Vector Math 30; color Mix 19. Noise supports
1D–4D and all five fractal modes; Voronoi supports all five features, four
distance metrics and normalization. Curves and color ramps use 257-sample
tables, so narrow/discontinuous features can differ from analytical Cycles.

Principled graph outputs include Base Color, Roughness, Metallic, IOR,
Transmission Weight, Emission Color/Strength, Alpha, Specular IOR Level,
Anisotropy/Rotation, Coat Weight/Roughness/IOR, Sheen Weight/Roughness/Tint,
Subsurface Weight/Radius, Normal and Tangent. Extra Principled inputs such as
coat/specular tint, coat normal, thin film and detailed SSS controls remain
unsupported. Diffuse roughness, exact sheen/hair/SSS models, refraction-only
closure behavior and colored transparency are not Cycles-equivalent.

Image textures are sampled at hits and graph-driven emission also participates
in mesh-light importance sampling. A shader tree and all materials on one mesh
must agree on one UV and one color layer. Missing attributes and animated image
users are not a general-purpose Blender attribute/texture system.

## Complete inventory

Native: **45**, Partial: **24**, Legacy: **8**, Unsupported: **17**, Outside Cycles surface scope: **5**.

| Blender node | Status | Scope |
|---|---|---|
| `ShaderNodeAddShader` | Native | Separate weighted closures, including dynamic factors; 16 leaves / 8 levels maximum. |
| `ShaderNodeAmbientOcclusion` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeAttribute` | Partial | One mesh color attribute: Color/Vector, average Fac, Alpha. Arbitrary scalar/vector, object, instancer and volume attributes are unsupported. |
| `ShaderNodeBackground` | Legacy | World export and native sky / bake fallback; not the surface interpreter. |
| `ShaderNodeBevel` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeBlackbody` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeBrightContrast` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeBsdfAnisotropic` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfDiffuse` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfGlass` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfHair` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfHairPrincipled` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeBsdfMetallic` | Partial | F82 base color, roughness and anisotropy mapped to vibrt metal. Physical conductor, edge tint and thin film are unsupported. |
| `ShaderNodeBsdfPrincipled` | Partial | Dynamic base/roughness/metallic/IOR/transmission/alpha/emission/specular/anisotropy/coat/sheen/SSS inputs listed below. Renderer closure physics are not identical to Cycles. |
| `ShaderNodeBsdfRayPortal` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeBsdfRefraction` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfSheen` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfToon` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeBsdfTranslucent` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfTransparent` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBump` | Partial | A Bump connected directly to a material Normal uses finite spatial differences. Nested Bump and displacement-aware derivatives are unsupported. |
| `ShaderNodeCameraData` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeClamp` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeCombineColor` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeCombineXYZ` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeDisplacement` | Legacy | Existing mesh texture displacement path; no general shader displacement evaluation. |
| `ShaderNodeEeveeSpecular` | Outside Cycles surface scope | EEVEE, Freestyle or legacy node; not claimed as Cycles-compatible. |
| `ShaderNodeEmission` | Native | Dynamic color and strength on hits and mesh-light samples. |
| `ShaderNodeFloatCurve` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeFresnel` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeGamma` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeGroup` | Native | Nested groups and instance defaults; reroutes and muted value nodes are followed. |
| `ShaderNodeHairInfo` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeHoldout` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeHueSaturation` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeInvert` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeLayerWeight` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeLightFalloff` | Partial | Surface-context ray length; light datablock graphs still use the existing light exporter. |
| `ShaderNodeLightPath` | Partial | Common ray flags, length, total/diffuse/glossy/transmission depth. Shadow/light-sample depth and volume/transparent/portal path state are incomplete. |
| `ShaderNodeMapRange` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeMapping` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeMath` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeMix` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeMixRGB` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeMixShader` | Native | Separate weighted closures, including dynamic factors; 16 leaves / 8 levels maximum. |
| `ShaderNodeNewGeometry` | Partial | Position, shading/geometric normal, incoming, tangent, backfacing. No pointiness or random-per-island attributes. |
| `ShaderNodeNormal` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeNormalMap` | Partial | Tangent/object/world and legacy Blender spaces; OpenGL/DirectX. Uses triangle UV derivatives, not a full exported MikkTSpace frame; displaced-base selection is not available. |
| `ShaderNodeObjectInfo` | Partial | Location, color, alpha, object/material pass indices; Random uses vibrt instance hashing, not Cycles hash identity. |
| `ShaderNodeOutputAOV` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeOutputLight` | Legacy | Existing light/IES exporter and photometric sampling; no general light shader DAG. |
| `ShaderNodeOutputLineStyle` | Outside Cycles surface scope | EEVEE, Freestyle or legacy node; not claimed as Cycles-compatible. |
| `ShaderNodeOutputMaterial` | Native | Surface dispatch; Volume and Displacement retain their documented legacy paths. |
| `ShaderNodeOutputWorld` | Legacy | World export and native sky / bake fallback; not the surface interpreter. |
| `ShaderNodeParticleInfo` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodePointInfo` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeRGB` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeRGBCurve` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeRGBToBW` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeRadialTiling` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeRaycast` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeScript` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeSeparateColor` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeSeparateXYZ` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeShaderToRGB` | Outside Cycles surface scope | EEVEE, Freestyle or legacy node; not claimed as Cycles-compatible. |
| `ShaderNodeSqueeze` | Outside Cycles surface scope | EEVEE, Freestyle or legacy node; not claimed as Cycles-compatible. |
| `ShaderNodeSubsurfaceScattering` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeTangent` | Partial | UV derivative or radial tangent; no MikkTSpace attribute export. |
| `ShaderNodeTexBrick` | Native | Spatial evaluation with linked inputs; Cycles 5.1.1 numerical helpers. |
| `ShaderNodeTexChecker` | Native | Spatial evaluation with linked inputs; Cycles 5.1.1 numerical helpers. |
| `ShaderNodeTexCoord` | Partial | Generated, UV, object, normal, camera, window, reflection. No explicit reference object or instancer coordinates; generated coordinates use evaluated mesh bounds. |
| `ShaderNodeTexEnvironment` | Partial | Flat/box/sphere/tube or environment projection, RGBA, extension and interpolation. No UDIM texture manager; animated image frames are not synchronized per image-user. |
| `ShaderNodeTexGabor` | Native | Spatial evaluation with linked inputs; Cycles 5.1.1 numerical helpers. |
| `ShaderNodeTexGradient` | Native | Spatial evaluation with linked inputs; Cycles 5.1.1 numerical helpers. |
| `ShaderNodeTexIES` | Legacy | Existing light/IES exporter and photometric sampling; no general light shader DAG. |
| `ShaderNodeTexImage` | Partial | Flat/box/sphere/tube or environment projection, RGBA, extension and interpolation. No UDIM texture manager; animated image frames are not synchronized per image-user. |
| `ShaderNodeTexMagic` | Native | Spatial evaluation with linked inputs; Cycles 5.1.1 numerical helpers. |
| `ShaderNodeTexNoise` | Native | Spatial evaluation with linked inputs; Cycles 5.1.1 numerical helpers. |
| `ShaderNodeTexSky` | Partial | Preetham, Hosek-Wilkie, Nishita, and multiple-scattering models; material sky uses a 2048×1024 native LUT. World uses the existing native-sky path or Cycles bake fallback. |
| `ShaderNodeTexVoronoi` | Native | Spatial evaluation with linked inputs; Cycles 5.1.1 numerical helpers. |
| `ShaderNodeTexWave` | Native | Spatial evaluation with linked inputs; Cycles 5.1.1 numerical helpers. |
| `ShaderNodeTexWhiteNoise` | Native | Spatial evaluation with linked inputs; Cycles 5.1.1 numerical helpers. |
| `ShaderNodeUVAlongStroke` | Outside Cycles surface scope | EEVEE, Freestyle or legacy node; not claimed as Cycles-compatible. |
| `ShaderNodeUVMap` | Partial | One named UV stream per mesh. Conflicting named streams fail explicitly; instancer UV unsupported. |
| `ShaderNodeValToRGB` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeValue` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeVectorCurve` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeVectorDisplacement` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeVectorMath` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeVectorRotate` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeVectorTransform` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeVertexColor` | Partial | Selected or active color layer, including Alpha; one color stream per mesh. |
| `ShaderNodeVolumeAbsorption` | Legacy | Existing homogeneous volume export. No general spatial volume graph. |
| `ShaderNodeVolumeCoefficients` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeVolumeInfo` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |
| `ShaderNodeVolumePrincipled` | Legacy | Existing homogeneous volume export. No general spatial volume graph. |
| `ShaderNodeVolumeScatter` | Legacy | Existing homogeneous volume export. No general spatial volume graph. |
| `ShaderNodeWavelength` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeWireframe` | Unsupported | Requires additional closure, attribute, traversal, volume or output integration. Export diagnostics/fallback do not constitute native support. |

## Validation

- `cargo test -p vibrt-nodes`: validation, reference fixtures and ABI boundaries.
- `make test`: struct layouts and Python exporter/adapter regressions.
- `blender -b --factory-startup --python-exit-code 1 --python scripts/test_shader_nodes.py`:
  315 operation/mode cases compared with Cycles and actual CUDA renders.
- The same command with `scripts/test_shader_graphs.py`: spatial images,
  48 cases covering named UV/color layers, alpha, object data, normal maps,
  tangents, group instance defaults and Mix/Add.
- `scripts/test_shader_lighting.py`: procedural emission illuminating a diffuse
  receiver, compared with a 512-sample Cycles render.
- `scripts/inventory_shader_nodes.py -- target/node-inventory.json`: regenerate
  the installed Blender node inventory (run through Blender).

Test artifacts are under `target/shader-nodes` and `target/shader-graphs`.
Pinned source, hashes, adaptations and licenses are documented in
[`vibrt-nodes/third_party/cycles`](../vibrt-nodes/third_party/cycles/README.md).
