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
compile; measure that separately from per-sample rendering time.

Packing removes unreachable nodes and reuses temporary slots after their last
read. The per-scene kernel reserves only the required slot capacity, rounded
to a power of two. Bump derivatives evaluate height dependencies; shadow rays
evaluate opacity; light samples evaluate emission and opacity. Coordinate and
matrix preparation is limited to contexts actually read by each graph.

Identical graph programs share one GPU upload within a render. Adjacent BSDF
closures can reuse the last graph's values and bumped normal at the same
intersection; the cache resets for every surface evaluation. Fully opaque
Mix/Add trees skip graph opacity evaluation on shadow rays. Texture addressing
avoids integer division for in-range coordinates and power-of-two wrapping,
while preserving repeat, mirror, extend and clip behavior.

Surface preparation combines input requirements across the complete Mix/Add
tree. UV derivatives, generated coordinates, object normals and vertex colors
are calculated only when used; triangle areas are needed only for mesh-light
MIS. Bump evaluations offset the required coordinate inputs without copying
the entire intersection record.

Single-output constant, UV and flat UV-image graphs (including an optional
socket conversion) use one packed instruction and bypass the GPU interpreter.
They share its texture filtering, alpha handling and non-finite-value cleanup.
Projected images and more complex graphs retain the general evaluator.

Before render setup or shader precompilation, pure subgraphs with constant
inputs are evaluated once by the shared C++ CPU implementation. This includes
math, color/vector operations, curves/ramps and procedurals with constant
coordinates. Unreachable nodes are removed and bit-identical float4 constants
are shared before kernel requirements and LUT uploads are computed. Image,
shading-context and scene-query dependencies remain dynamic. The CPU evaluator
used for displacement applies the same optimization; alpha, HDR values and
per-operation non-finite cleanup follow the shader VM.

Math implements 41 operations; Vector Math 30; color Mix 19. Noise supports
1D–4D and all five fractal modes; Voronoi supports all five features, four
distance metrics and normalization. Curves and color ramps use 257-sample
tables, so narrow/discontinuous features can differ from analytical Cycles.

Principled graph outputs include Base Color, Roughness, Metallic, IOR,
Transmission Weight, Emission Color/Strength, Alpha, Specular IOR Level,
Anisotropy/Rotation, Coat Weight/Roughness/IOR, Sheen Weight/Roughness/Tint,
Subsurface Weight/Radius, Normal and Tangent. Extra Principled inputs such as
coat/specular tint, coat normal, thin film and detailed SSS controls remain
unsupported. Diffuse roughness, exact sheen, the legacy Hair BSDF, Principled SSS,
refraction-only closure behavior and colored transparency are not Cycles-equivalent.
Standalone Chiang Hair and Burley SSS use the additional closures described below.

Image textures are sampled at hits and graph-driven emission also participates
in mesh-light importance sampling. Named UV/color/scalar/vector attributes are exported only when requested, and
several layers can coexist. Animated image users and UDIM tiles still require a
texture manager. Default UV images reuse the existing coordinate buffer.

## Scene-dependent additions

AO, Bevel, Raycast and Burley SSS use compact per-mesh/instance software BVHs;
ordinary path tracing continues to use OptiX. These extra BVHs are allocated only
when needed. AOV and extended volume-phase kernels are also compiled only for
scenes using them. Device uploads are owned by each render and released on
success, cancellation or errors.

Geometric displacement runs the shared C++ evaluator on the CPU before upload,
borrowing image buffers. It edits only evaluated mesh copies, averages corner
samples at shared vertices and uses the available tessellation. Ray-dependent,
scene-query, derivative and instance-property displacement inputs are rejected.
Displacement is evaluated on source objects before instancing. BOTH uses geometric
displacement without a second fine-detail bump pass. BUMP updates the closure
normal; Geometry Normal read elsewhere in the graph is still undisplaced.

The remaining unsupported node types are **Point Info, Volume Info and Script
(OSL)**. They need point-cloud geometry, a volume-grid backend and an OSL runtime,
respectively. Huang Hair and random-walk SSS remain unsupported modes; Huang's
prototype failed roughness-dependent Cycles comparisons and is not enabled.
Warnings/fallbacks do not count as implementation of these features.

## Complete inventory

Native: **52**, Partial: **32**, Legacy: **7**, Unsupported: **3**, Outside Cycles surface scope: **5**.

| Blender node | Status | Scope |
|---|---|---|
| `ShaderNodeAddShader` | Native | Separate weighted closures, including dynamic factors; 16 leaves / 8 levels maximum. |
| `ShaderNodeAmbientOcclusion` | Native | Color/AO outputs, distance, samples, inside and local-only queries against evaluated mesh geometry. |
| `ShaderNodeAttribute` | Partial | Named POINT/CORNER/FACE scalar, integer, boolean, vector and color attributes; object/instancer/view-layer custom properties. Volume grids remain unsupported. |
| `ShaderNodeBackground` | Legacy | World export and native sky / bake fallback; not the surface interpreter. |
| `ShaderNodeBevel` | Partial | Cycles cubic profile and three-axis sampling on triangle geometry; at most 64 intersections per sample. |
| `ShaderNodeBlackbody` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeBrightContrast` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeBsdfAnisotropic` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfDiffuse` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfGlass` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfHair` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfHairPrincipled` | Partial | Unmodified Cycles Chiang closure with Color/Melanin/Absorption parametrizations and strand randomization. Huang is rejected pending agreement with Cycles. |
| `ShaderNodeBsdfMetallic` | Partial | F82 base color, roughness and anisotropy mapped to vibrt metal. Physical conductor, edge tint and thin film are unsupported. |
| `ShaderNodeBsdfPrincipled` | Partial | Dynamic base/roughness/metallic/IOR/transmission/alpha/emission/specular/anisotropy/coat/sheen/SSS inputs listed below. Renderer closure physics are not identical to Cycles. |
| `ShaderNodeBsdfRayPortal` | Native | Dynamic color, position and direction; ray relocation also works inside Mix/Add trees. |
| `ShaderNodeBsdfRefraction` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfSheen` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfToon` | Native | Diffuse and glossy components with dynamic Color, Size, Smooth and Normal. |
| `ShaderNodeBsdfTranslucent` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBsdfTransparent` | Partial | Existing vibrt closure with dynamic supported color/value/normal sockets. See closure limitations below. |
| `ShaderNodeBump` | Partial | Nested Bump graphs and normal chains use finite spatial differences. Ray differentials and displacement-aware Geometry Normal are incomplete. |
| `ShaderNodeCameraData` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeClamp` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeCombineColor` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeCombineXYZ` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeDisplacement` | Partial | Shared C++ evaluator for geometric DISPLACEMENT/BOTH on evaluated mesh vertices and BUMP shading. No adaptive subdivision; see limits below. |
| `ShaderNodeEeveeSpecular` | Outside Cycles surface scope | EEVEE, Freestyle or legacy node; not claimed as Cycles-compatible. |
| `ShaderNodeEmission` | Native | Dynamic color and strength on hits and mesh-light samples. |
| `ShaderNodeFloatCurve` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeFresnel` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeGamma` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeGroup` | Native | Nested groups and instance defaults; reroutes and muted value nodes are followed. |
| `ShaderNodeHairInfo` | Partial | Is Strand, Intercept, Length, Thickness, Random and Tangent Normal on exported legacy particle-hair ribbons. Modern Curves geometry is not exported. |
| `ShaderNodeHoldout` | Native | Black contribution and holdout alpha with transparent film; weighted Mix/Add support. |
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
| `ShaderNodeNormalMap` | Partial | Multiple explicitly named UV maps use exported MikkTSpace tangents. The default unnamed UV retains triangle-derivative tangents; object/world and OpenGL/DirectX are supported. |
| `ShaderNodeObjectInfo` | Partial | Location, color, alpha, object/material pass indices; Random uses vibrt instance hashing, not Cycles hash identity. |
| `ShaderNodeOutputAOV` | Native | Color/Value roots, including nested groups, written to registered Blender view-layer passes on render completion. |
| `ShaderNodeOutputLight` | Legacy | Existing light/IES exporter and photometric sampling; no general light shader DAG. |
| `ShaderNodeOutputLineStyle` | Outside Cycles surface scope | EEVEE, Freestyle or legacy node; not claimed as Cycles-compatible. |
| `ShaderNodeOutputMaterial` | Native | Surface and AOV dispatch; Volume and Displacement follow the limitations documented for those nodes. |
| `ShaderNodeOutputWorld` | Legacy | World export and native sky / bake fallback; not the surface interpreter. |
| `ShaderNodeParticleInfo` | Partial | Index, age, lifetime, location, size, velocity, angular velocity and random on legacy particle object instances. Missing particle data returns zero. |
| `ShaderNodePointInfo` | Unsupported | Requires native point-cloud geometry export and intersection data; rejected explicitly. |
| `ShaderNodeRGB` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeRGBCurve` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeRGBToBW` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeRadialTiling` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeRaycast` | Native | Is Hit, Is Hit Self, Distance, Position and Normal; scene/local query modes on evaluated triangles. |
| `ShaderNodeScript` | Unsupported | OSL shaders require an OSL execution backend and renderer services; rejected explicitly. |
| `ShaderNodeSeparateColor` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeSeparateXYZ` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeShaderToRGB` | Outside Cycles surface scope | EEVEE, Freestyle or legacy node; not claimed as Cycles-compatible. |
| `ShaderNodeSqueeze` | Outside Cycles surface scope | EEVEE, Freestyle or legacy node; not claimed as Cycles-compatible. |
| `ShaderNodeSubsurfaceScattering` | Partial | Burley disk scattering with Cycles radial profiles and three-axis MIS. Random Walk and Random Walk Skin are rejected; intersection enumeration is capped at 64. |
| `ShaderNodeTangent` | Partial | Named/active UV MikkTSpace tangents and radial tangents on meshes. Curve tangent semantics retain the ribbon approximation. |
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
| `ShaderNodeUVMap` | Partial | Multiple independently named UV layers per mesh; instancer UV remains unsupported. |
| `ShaderNodeValToRGB` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeValue` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeVectorCurve` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeVectorDisplacement` | Partial | Tangent/object/world vectors, midlevel and scale; geometric displacement or BUMP shading. No adaptive subdivision. |
| `ShaderNodeVectorMath` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeVectorRotate` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeVectorTransform` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeVertexColor` | Native | Multiple named/active mesh color layers, including Alpha, in POINT or CORNER domains. |
| `ShaderNodeVolumeAbsorption` | Legacy | Existing homogeneous volume export. No general spatial volume graph. |
| `ShaderNodeVolumeCoefficients` | Partial | Homogeneous scattering/absorption/emission with HG, Fournier-Forand, Draine, Rayleigh and Mie phase functions. Spatial graphs/grids and mixed distinct phase functions are unsupported. |
| `ShaderNodeVolumeInfo` | Unsupported | Requires density/color/flame/temperature grid sampling and heterogeneous volume transport; rejected explicitly. |
| `ShaderNodeVolumePrincipled` | Legacy | Existing homogeneous volume export. No general spatial volume graph. |
| `ShaderNodeVolumeScatter` | Legacy | Existing homogeneous volume export. No general spatial volume graph. |
| `ShaderNodeWavelength` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeWireframe` | Partial | Triangle edges with world or pixel size. Pixel size uses the primary camera footprint rather than propagated ray differentials. |

## Validation

- `cargo test -p vibrt-nodes`: validation, reference fixtures and ABI boundaries.
- `cargo test -p vibrt --lib`: material-tree input requirements, opaque-tree
  classification and child fallbacks
  (requires the native renderer build dependencies).
- `make test`: struct layouts and Python exporter/adapter regressions.
- `blender -b --factory-startup --python-exit-code 1 --python scripts/test_shader_nodes.py`:
  315 operation/mode cases compared with Cycles and actual CUDA renders.
- The same command with `scripts/test_shader_graphs.py`: spatial images,
  65 cases covering multiple UV/color/scalar/vector layers, object/instancer/layer
  properties, normal maps, nested bump, displacement values and Mix/Add.
- `scripts/test_scene_nodes.py`: 18 Cycles comparisons for AO, Bevel, Wireframe
  and Raycast, including local queries, self hits and misses.
- `scripts/test_extended_closures.py`: Toon, Holdout, Ray Portal, Chiang's three
  parametrizations and Burley SSS, with Color/Value AOV readback.
- `scripts/test_aov_passes.py`: actual Blender RenderResult Color/Value passes
  and transparent-film alpha.
- `scripts/test_displacement.py`: tangent/object/world geometric displacement
  and image heights, BUMP isolation and unsupported-input checks, including
  preservation of the original Blender mesh.
- `scripts/test_node_exports.py`: real particle-instance and hair metadata,
  volume-phase export and explicit unsupported-mode diagnostics.
- `scripts/test_medium_closures.py`: curved Burley SSS and five homogeneous
  volume phases compared with Cycles at 2048 samples.
- `scripts/test_shader_lighting.py`: procedural emission illuminating a diffuse
  receiver, compared with a 512-sample Cycles render.
- `scripts/test_bsdf_mixtures.py`: nine Mix/Add energy checks against separately
  rendered components, including tilted normals, coat, glass, alpha, hair,
  translucency, nested mixtures and zero/full weights. Run through Blender
  with the same headless command above.
- `scripts/inventory_shader_nodes.py -- target/node-inventory.json`: regenerate
  the installed Blender node inventory (run through Blender).

Test artifacts are under `target/shader-nodes`, `target/shader-graphs`,
`target/scene-nodes`, `target/extended-closures`, `target/medium-closures`,
`target/shader-lighting` and `target/bsdf-mixtures`.
Pinned source, hashes, adaptations and licenses are documented in
[`vibrt-nodes/third_party/cycles`](../vibrt-nodes/third_party/cycles/README.md).

## Rendering performance

In Blender's **Render Properties > Shaders**, click **Precompile Shaders** to
prepare the current scene/view-layer shader variant before rendering. The
background job fills both the PTX and OptiX caches using the same compiler as
the renderer. Progress and a Cancel button appear in the panel; Esc also stops
the job. Unsaved edits are included in a temporary snapshot, without replacing
the current Render Result or saving the working `.blend` file.

This prepares shader compilation only. Scene export, texture uploads and BVH
construction still happen when rendering. Changing node types, closure trees,
AOVs or volume phases can require a different variant; run the button again
after those edits. Material discovery uses a render-mode dependency graph and
skips mesh export and world-lighting preparation.

Run the same scene, resolution, samples and preview mode for each revision:

```sh
blender -b --factory-startup test_scenes/classroom/classroom/classroom.blend \
  --python-exit-code 1 --python scripts/benchmark_render.py -- \
  --output target/render-bench/current --percentage 25 --spp 32
```

The script exports once and renders twice without denoising. `results.json`
separates export/preparation from render time and records preview callbacks
at the addon's default 100 ms cadence, without Blender display work.
`--preview-interval-ms 0` measures explicit every-sample display;
`--preview-delay-ms 30` simulates a 30 ms synchronous display callback.
`--repo` selects another checkout with its own
staged native extension; `--batch` disables intermediate readback and sample
callbacks. Normal and headless renders queue short batches of one-sample launches;
explicit every-sample previews wait after each sample. Compare the same cadence
and callback delay across revisions. Callback timings average over the samples
between snapshots; they do not measure individual GPU launches. Simulated delays
are not measurements of Blender display work. Older checkouts may have used a
different launch schedule.
`--without-graphs` and `--without-bump` are diagnostic ablations that change
materials. Likewise, versions predating graph support can omit material detail;
their timings do not represent equivalent rendering quality.
