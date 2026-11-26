# vibrt-nodes

Sky, shader value nodes and surface closures, independent of Blender's runtime.
Rust owns buffers, validates inputs and packs graphs; C++ evaluates node values.

- `src/graph.rs`, `src/types.rs`: scene-facing graph schema and GPU layout.
- `native/core.h`: one color implementation compiled for CPU batches and CUDA.
- `native/color_bridge.cpp`: the synchronous C ABI used by Rust's `color::apply`.
- `src/sky.rs`, `native/sky_bridge.cpp`, `third_party/sky/`: atmosphere settings,
  ownership and Blender 5.1.1's sky models.
- `src/fold.rs`: constant propagation using the same C++ color operations.
- `src/shader.rs`: validated multi-output shader DAG, texture/LUT bindings.
- `native/shader.h`: shared CPU/CUDA interpreter for typed scalar, vector and
  color sockets, image/procedural textures, normals and shading context.
- `native/bsdf.h`: Cycles closure assembly from explicit material parameters.
- `third_party/blender/`: Blender v5.1.1 git submodule; `build.rs` extracts the
  numerical functions and tables from `intern/cycles` without modifying them.
- `third_party/cycles/README.md`: source provenance, adapters and license notes.

The Blender addon reads node connections and curve tables. Its `node_ops.py`
adapter normalizes array shapes and calls the PyO3 bindings in
`vibrt/src/python_nodes.rs`. No borrowed Blender/NumPy pointers survive a call.
CPU workers read immutable inputs while the caller retains the Python GIL.

The renderer uploads graph buffers and embeds `DEVICE_TYPES` / `DEVICE_CORE`
in its NVRTC source. Image sampling, vertex color and object random use the
shading inputs supplied by the renderer. General shader graphs use 104-byte
instructions and up to 256 value nodes, with explicit output sockets. The
older color-only graph remains compatible (64-byte records, 32 nodes).
Oversized graphs, invalid modes, dangling links and cycles are rejected.
No source files or C++ compiler are needed by an installed addon at runtime.

General graphs use Blender socket conversion rules and evaluate each connected
material parameter at the hit point. Mix/Add Shader retain separate closures.
Closure evaluation, sampling and PDFs reuse Cycles C++ on CPU/CUDA. Principled
layer assembly adapts Cycles' SVM setup to explicit parameters; scene traversal,
light sampling and path integration remain renderer responsibilities.
`ShaderGraph::optimized()` folds pure constant subgraphs through the C++ VM,
prunes unused nodes and shares identical expressions, including repeated image,
noise and coordinate calculations. Keys include all float bits, resource IDs,
attributes and derivative offsets. Values still update at every shading point;
traced scene queries remain separate. Rendering and shader precompile
apply it before choosing kernels and uploading graphs; CPU graph evaluation
also uses it.
See the [complete support matrix](../docs/shader-nodes.md) for supported modes
and remaining renderer-specific limitations. The older color path preserves
its scalar-folding and per-channel image Math conventions.
See [sky provenance and licenses](third_party/sky/README.md) for the reused
Blender atmosphere sources.

## Validation

Requires Rust and a C++17 compiler; the core needs no Blender, CUDA or OptiX:

```sh
git submodule update --init --depth 1
cargo test -p vibrt-nodes
```

Fixtures record the pre-extraction Python results for 25 graph-fold cases and
222 scalar Math cases. Tests also cover broadcasting, HDR curves, invalid
inputs, graph references and stack limits.

With the renderer's CUDA/OptiX development environment and an NVIDIA GPU:

```sh
cargo run --release -p vibrt --example check_nodes
```

This compares 37 color-node cases across 4,096 RGB inputs each using C++ on
the CPU and the graph interpreter on CUDA. After building/staging the Python
extension, `scripts/test_native_sky.py` checks the sky models against Cycles,
and `scripts/test_render_job.py -- --expect-sky --expect-native-sky
--expect-progress` exercises F12 in a separate foreground-capable Blender.

`scripts/test_shader_nodes.py` compares numerical node cases against Cycles and
CUDA in one atlas. `scripts/test_shader_graphs.py` exercises images, named UVs,
vertex alpha, object attributes, normal maps, group instance defaults and
Mix/Add closures against Cycles. Run these through Blender with
`-b --factory-startup --python-exit-code 1 --python SCRIPT`.
`scripts/test_shader_lighting.py` checks illumination from procedural emissive
meshes, not just camera-visible emission. Pass `-- --layered` to check attenuation
by Principled coat and sheen at sampled light points.
`scripts/test_surface_closures.py` compares rough Diffuse, both Sheen modes,
GGX/Beckmann/Multi-GGX, physical/F82 metals, thin film, and linked Principled
layers under environment and Sun lighting.
