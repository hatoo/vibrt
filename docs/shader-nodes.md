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
coordinates. Unreachable nodes are removed and identical expressions are shared
before kernel requirements and LUT uploads are computed. This also avoids
repeating dynamic image, noise and coordinate calculations at a shading point;
keys preserve float bits, texture/attribute identity and derivative offsets.
Scene queries remain separate, and context-dependent values stay dynamic. The CPU
evaluator used for displacement applies the same optimization; alpha, HDR values and
per-operation non-finite cleanup follow the shader VM.

Math implements 41 operations; Vector Math 30; color Mix 19. Noise supports
1D–4D and all five fractal modes; Voronoi supports all five features, four
distance metrics and normalization. Curves and color ramps use 257-sample
tables, so narrow/discontinuous features can differ from analytical Cycles.

Principled graph outputs include Base Color, Roughness, Diffuse Roughness, Metallic, IOR,
Transmission Weight, Emission Color/Strength, Alpha, Specular IOR Level,
Anisotropy/Rotation, Coat Weight/Roughness/IOR/Tint/Normal, Specular Tint,
Sheen Weight/Roughness/Tint, Thin Film Thickness/IOR,
Subsurface Weight/Radius/Scale/IOR/Anisotropy, Normal and Tangent.
Surface evaluation, sampling and PDFs reuse Cycles' rough Diffuse, Sheen/Velvet
and GGX/Beckmann closures. Principled assembles sheen, coat, metal, transmission,
specular and diffuse/SSS in Cycles order, including attenuation of emission.
Metallic supports F82 Edge Tint and physical conductor IOR/Extinction.
Thin-film tables use scene-linear Rec.709. Legacy Hair reflection/transmission
and diffuse/glossy Toon use the upstream closures. Transparent closures preserve
RGB throughput on camera and shadow paths, including weighted Mix/Add trees. Standalone and
Principled SSS support Burley, Random Walk and Random Walk Skin. Random walks
use Cycles coefficient/radius fits and entry sampling, with spectral free flights,
forward/backward Dwivedi guiding and Russian roulette (1,024-event cap).

Map Range, Mapping and Vector Rotate call the upstream SVM functions, including
degenerate input ranges and zero rotation axes. IES parsing and interpolation
use Cycles for Type A/B/C profiles. Path samples use Owen-scrambled Sobol, and
surface/shadow origins use scale-aware ray offsets. Triangle lights use Cycles'
solid-angle/area sampling with matching hit PDFs. A spatial light tree selects
rectangular lights and mesh-light groups using Cycles orientation bounds and
SAOH partitioning; each mesh retains its triangle CDF. Point, Spot, Sun and Area
samplers, area spread and environment-map sampling/PDFs also call upstream code.

Closest, linear and cubic image interpolation share Cycles CPU kernels on both
CPU and CUDA, with packed texture storage supplied by the adapter. Graph-driven emission also participates
in mesh-light importance sampling. Named UV/color/scalar/vector attributes are exported only when requested, and
several layers can coexist. Animated image users and UDIM tiles still require a
texture manager. Default UV images reuse the existing coordinate buffer.
Unconnected hidden Normal inputs retain the surface normal through groups and
reroutes. Finite Point/Spot lights use sphere or facing-disk sampling for soft
shadows; these lights are still NEE-only, so direct visibility and delta-mirror
light hits remain unsupported.

## Scene-dependent additions

AO, Bevel, Raycast, SSS and initial volume containment share the OptiX acceleration
structures used by ordinary path tracing. A dedicated geometry-query ray type
preserves local-instance and self-hit filtering without applying material alpha.
No separate software BVH is built, uploaded or cached. The material evaluators, interpreter and
query service use OptiX continuation callables to bound code expansion when
shaders trace rays. NVRTC retains their entry points with relocatable device code.
OptiX uses the default optimization level.
AOV and extended volume-phase kernels are compiled only for
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
respectively. Chiang and Huang Hair use the upstream closures, including Huang
aspect ratio and R/TT/TRT lobes. Particle hair reads Blender's evaluated parent/child
cache, and modern Curves exports point/curve attributes. Both use OptiX round
linear segments with per-key radius; Catmull-Rom smoothing and motion keys are
not implemented, so sparse curved guides can differ from Cycles.
Warnings/fallbacks do not count as implementation of these features.

## Complete inventory

Native: **52**, Partial: **33**, Legacy: **6**, Unsupported: **3**, Outside Cycles surface scope: **5**.

| Blender node | Status | Scope |
|---|---|---|
| `ShaderNodeAddShader` | Native | Separate weighted closures, including dynamic factors; 16 leaves / 8 levels maximum. |
| `ShaderNodeAmbientOcclusion` | Native | Color/AO outputs, distance, samples, inside and local-only queries against evaluated mesh geometry. |
| `ShaderNodeAttribute` | Partial | Named POINT/CORNER/FACE scalar, integer, boolean, vector and color attributes; object/instancer/view-layer custom properties. Volume grids remain unsupported. |
| `ShaderNodeBackground` | Legacy | World export and native sky / bake fallback; not the surface interpreter. |
| `ShaderNodeBevel` | Partial | Cycles cubic profile and three-axis sampling on triangle geometry; at most 64 intersections per sample. |
| `ShaderNodeBlackbody` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeBrightContrast` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeBsdfAnisotropic` | Partial | Cycles GGX, Beckmann and Multi-GGX, with standalone anisotropy remapping and generated/linked tangent. |
| `ShaderNodeBsdfDiffuse` | Partial | Cycles Lambert and Oren-Nayar, including linked roughness, color and normal. |
| `ShaderNodeBsdfGlass` | Partial | Cycles GGX, Beckmann and Multi-GGX dielectric reflection/transmission with thin film. |
| `ShaderNodeBsdfHair` | Native | Cycles reflection/transmission closures; dynamic Color, Offset, RoughnessU/V and Tangent. Unlinked tangents follow strands or mesh dPdv. |
| `ShaderNodeBsdfHairPrincipled` | Partial | Unmodified Cycles Chiang/Huang closures, Color/Melanin/Absorption parametrizations, Huang aspect ratio and R/TT/TRT weights. Native round linear curves; no curve motion blur. |
| `ShaderNodeBsdfMetallic` | Partial | Cycles GGX/Multi-GGX with F82 Edge Tint or physical conductor IOR/Extinction, anisotropy and thin film; Rec.709 working space. |
| `ShaderNodeBsdfPrincipled` | Partial | Cycles numerical closures and layer ordering; dynamic tint, coat normal and thin film in addition to base/specular/coat/sheen/SSS inputs. Includes linked Diffuse Roughness; path integration remains renderer-specific. |
| `ShaderNodeBsdfRayPortal` | Native | Dynamic color, position and direction; ray relocation also works inside Mix/Add trees. |
| `ShaderNodeBsdfRefraction` | Partial | Cycles GGX/Beckmann refraction-only closure, including total internal reflection rejection. |
| `ShaderNodeBsdfSheen` | Partial | Cycles Microfiber LTC and Ashikhmin Velvet, including linked Color/Roughness/Normal. Integrator limitations still apply. |
| `ShaderNodeBsdfToon` | Native | Diffuse and glossy components with dynamic Color, Size, Smooth and Normal. |
| `ShaderNodeBsdfTranslucent` | Partial | Cycles back-hemisphere Lambert closure with linked color and normal. |
| `ShaderNodeBsdfTransparent` | Partial | Colored transmission and RGB shadow extinction via Cycles surface transparency, including weighted Mix/Add trees and the transparent bounce limit. |
| `ShaderNodeBump` | Partial | Cycles SVM bump with compact ray/UV differentials, Filter Width and nested normal chains. Displacement-aware Geometry Normal remains incomplete. |
| `ShaderNodeCameraData` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeClamp` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeCombineColor` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeCombineXYZ` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeDisplacement` | Partial | Shared C++ evaluator for geometric DISPLACEMENT/BOTH on evaluated mesh vertices and BUMP shading. No adaptive subdivision; see limits below. |
| `ShaderNodeEeveeSpecular` | Outside Cycles surface scope | EEVEE, Freestyle or legacy node; not claimed as Cycles-compatible. |
| `ShaderNodeEmission` | Native | Dynamic color and strength on hits, mesh/area-light samples, and homogeneous volume emission driven by ray flags. Spatial volume graphs are unsupported. |
| `ShaderNodeFloatCurve` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeFresnel` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeGamma` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeGroup` | Native | Nested groups, shader sockets and instance defaults; reroutes and muted nodes are followed. |
| `ShaderNodeHairInfo` | Partial | Is Strand, Intercept, Length, Thickness, Random and Tangent Normal on evaluated particle hair and modern Curves with native round linear geometry. |
| `ShaderNodeHoldout` | Native | Black contribution and holdout alpha with transparent film; weighted Mix/Add support. |
| `ShaderNodeHueSaturation` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeInvert` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeLayerWeight` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeLightFalloff` | Partial | Surface and finite-light shader ray length, including Linear/Constant falloff and Smooth. Sun uses the legacy light exporter. |
| `ShaderNodeLightPath` | Partial | Common ray flags, length, total/diffuse/glossy/transmission depth. Transparent transmission preserves the original ray flags and has a separate bounce budget. Shadow/light-sample depth and volume/transparent/portal counters are incomplete. |
| `ShaderNodeMapRange` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeMapping` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeMath` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeMix` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeMixRGB` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeMixShader` | Native | Separate weighted closures, including dynamic factors; 16 leaves / 8 levels maximum. |
| `ShaderNodeNewGeometry` | Partial | Position, shading/geometric normal, incoming, tangent, backfacing and welded-mesh Pointiness. Random-per-island is unsupported. |
| `ShaderNodeNormal` | Native | Shared C++ CPU/CUDA evaluator; socket conversions are explicit. |
| `ShaderNodeNormalMap` | Partial | Cycles SVM normal-map evaluation; explicitly named UV maps use exported MikkTSpace tangents. The default unnamed UV retains triangle-derivative tangents; object/world and OpenGL/DirectX are supported. |
| `ShaderNodeObjectInfo` | Partial | Location, color, alpha, object/material pass indices; Random uses vibrt instance hashing, not Cycles hash identity. |
| `ShaderNodeOutputAOV` | Native | Color/Value roots, including nested groups, written to registered Blender view-layer passes on render completion. |
| `ShaderNodeOutputLight` | Partial | Point/Spot/Area Emission/Mix/Add trees run at NEE samples (and area-light hits), including Ray Length. Sun and IES use the existing exporter. Full light-object attributes and shadow-sample path depths are incomplete. |
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
| `ShaderNodeSubsurfaceScattering` | Partial | Burley disk scattering (64-intersection cap), Random Walk and Random Walk Skin interior transport; see sampling and closure limitations above. |
| `ShaderNodeTangent` | Partial | Named/active UV MikkTSpace tangents and radial tangents on meshes. Curve tangents follow the native segment. |
| `ShaderNodeTexBrick` | Native | Spatial evaluation with linked inputs; Cycles 5.1.1 numerical helpers. |
| `ShaderNodeTexChecker` | Native | Spatial evaluation with linked inputs; Cycles 5.1.1 numerical helpers. |
| `ShaderNodeTexCoord` | Partial | Generated, UV, object, normal, camera, window, reflection. Explicit reference objects use their inverse world transform; instancer coordinates remain unsupported. Generated coordinates use evaluated mesh bounds. |
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
| `ShaderNodeWireframe` | Partial | Cycles triangle-edge evaluation with world or pixel size, using propagated compact ray differentials. Curves return zero as in Cycles. |

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
  receiver, compared with a 512-sample Cycles render. `-- --many` exercises nine
  emitter groups and spatial-tree MIS.
- `scripts/test_blender_reuse.py`: Hair reflection/transmission, diffuse/glossy
  Toon and isotropic/anisotropic Random Walk compared with Cycles.
- `cargo test -p vibrt-nodes --test reuse`: IES A/B/C and malformed data, value-node
  edge cases, sample/evaluate PDF agreement, Sobol strata, ray offsets, Dwivedi
  normalization and light-tree probabilities.
- `scripts/check_device_compile.py`: CPU-only NVRTC compilation on Windows;
  `--minimal` checks a scene without special closures.
- `scripts/test_bsdf_mixtures.py`: nine Mix/Add energy checks against separately
  rendered components, including tilted normals, coat, glass, alpha, hair,
  translucency, nested mixtures and zero/full weights. Run through Blender
  with the same headless command above.
- `scripts/inventory_shader_nodes.py -- target/node-inventory.json`: regenerate
  the installed Blender node inventory (run through Blender).

Test artifacts are under `target/shader-nodes`, `target/shader-graphs`,
`target/scene-nodes`, `target/extended-closures`, `target/medium-closures`,
`target/shader-lighting`, `target/shader-lighting-many`, `target/blender-reuse-render`
and `target/bsdf-mixtures`.
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

`scripts/test_scene_parity.py` compares orthographic cameras and pixel aspect,
explicit object coordinates, Pointiness, missing images, shader group inputs,
implicit group normals, elliptical light shader graphs, camera-inside emission volumes,
transparent bounce budgets, Point/Spot falloff and soft shadows, and all three
SSS methods against Cycles. Outputs are ignored under `target/scene-parity-tests/`.

Surface closure regression: `scripts/test_surface_closures.py` renders 50 materials
under environment and Sun illumination with 8,192 samples and unclamped
contributions. It checks per-tile linear RGB against Cycles (8% relative,
0.04 dark-value floor), including linked tint/normal/film/conductor sockets.

Additional reuse regressions:

- `scripts/test_native_curves.py`: Chiang/Huang, circular/elliptical cross sections.
- `scripts/test_reuse_sampling.py`: finite/delta Point/Spot, scaled Spot, Sun,
  rectangular/disk/elliptical Area and spread, colored transparent/glass shadows,
  adaptive convergence and AOV normalization.
- `scripts/test_reuse_transport.py`: Box/Gaussian/Blackman-Harris film filters,
  circular/polygon/anamorphic apertures, orthographic DOF, and finite emitters
  inside a colored absorbing/scattering medium.

Film filters use Cycles inverse-CDF tables. Adaptive sampling uses the upstream
class-A estimator, convergence check and two neighbor filter passes, with actual
per-pixel counts for image/AOV normalization. Converged pixels skip work in later
launches; the render still iterates to the configured sample limit. Homogeneous
volume transport mixes Cycles equiangular and exponential distance proposals
when a finite emitter is available; heterogeneous grids remain unsupported.
