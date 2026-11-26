# vibrt

A GPU path-tracing renderer for Blender, powered by NVIDIA OptiX 9.

## Crates

| Crate | Description |
|-------|-------------|
| [`optix-sys`](optix-sys/) | Raw FFI bindings to OptiX 9.0.0 (bindgen) |
| [`optix`](optix/) | Safe Rust wrapper with RAII, builders, and type-safe enums |
| [`vibrt-nodes`](vibrt-nodes/) | Standalone shader and sky nodes with shared C++/CUDA evaluation |
| [`vibrt`](vibrt/) | OptiX renderer and `vibrt_native` Python extension |

The [`blender/vibrt_blender/`](blender/vibrt_blender/) addon registers `vibrt` as a render engine. It passes a live scene dictionary and NumPy buffers to the native extension in the same Blender process, then updates Blender's Render Result.

See [module boundaries and resource lifetimes](docs/architecture.md) for the renderer's session/scene/job API and the export context shared by sky and material preparation.

## Features

- Path tracing with direct-light sampling for point / sun / spot / area-rect lights, emissive meshes and HDRI environments; MIS for competing sampling paths
- Transparent shadow rays through transmissive / alpha-cutout surfaces
- GPU shader graphs for linked material inputs, image/procedural textures, color/vector math, and Mix/Add closures; see the [node support matrix](docs/shader-nodes.md) for supported modes and approximations
- GGX multi-scattering energy compensation (Kulla-Conty)
- Direct/indirect contribution clamping using the sum of absolute RGB components
- Triangle mesh geometry with multi-material slots (per-triangle material index); multi-object scenes via single-level OptiX instancing
- Right-handed Z-up (no axis conversion from Blender)
- Perspective camera with lens shift and circular-aperture depth of field
- Native Preetham, Hosek/Wilkie and single/multiple scattering sky models on Blender 5.1
- Homogeneous volumes, legacy particle-hair ribbons, Chiang hair, Burley and Random Walk SSS
- Progressive sample previews, cancellation, and optional final OptiX denoising
- Output: linear RGBA, depth and named AOV buffers delivered to Blender; Blender handles color management and image-file output

## Requirements

- **Rust** and **Python 3.11+** — for the native extension and build scripts
- **C++17 compiler** (MSVC, Clang or GCC) — for the shared node and atmosphere implementations
- **NVIDIA OptiX SDK 9.0.0** — set `OPTIX_ROOT` or install at default location
- **CUDA Toolkit** — for NVRTC runtime compilation
- **LLVM/Clang** — for bindgen (set `LIBCLANG_PATH` if not auto-detected)
- **NVIDIA GPU** with driver supporting OptiX 9
- **Blender 4.0+** (tested on 5.1) — for the addon

## Building

```bash
make dev-install          # build, stage, and link the addon into Blender
make native-build         # rebuild and stage the extension after native edits
make addon-with-native    # package blender/vibrt_blender.zip for installation
```

To build just the native library, run `cargo build --release -p vibrt --features python`. The addon needs the resulting extension staged beside its Python sources. `make addon` packages Python sources only. See [addon installation and usage](blender/README.md).

## Usage

Select **vibrt** as Blender's render engine and press **F12**. The first completed sample appears immediately; subsequent previews update periodically. **Render Properties → Shaders → Precompile Shaders** warms the shader caches before rendering.

There is no standalone renderer executable. [`scripts/render_blend.py`](scripts/render_blend.py) launches headless Blender, loads the working-tree addon and renders through the same native extension:

```bash
py scripts/render_blend.py scene.blend --output render.png --spp 128
```

## Tests and preview scenes

The [`Makefile`](Makefile) renders `.blend` fixtures. The IES scene is tracked under [`test_scenes/ies_light/`](test_scenes/ies_light/); large scenes such as classroom and junk_shop are local, gitignored assets. `previews` and `cycles-previews` require all scenes listed in `BLEND_SCENES`.

```bash
make test                              # layouts, Rust node tests, Python suites
make ies_light-preview SPP=32 PCT=50   # vibrt preview
make ies_light-cycles SPP=32 PCT=50    # Cycles reference
cargo fmt --all -- --check
cargo clippy --workspace --features python
deslop-please --features python       # module boundaries and dependencies
```

`make test` uses a POSIX shell and NumPy; native binding tests need a staged extension. `cargo test -p vibrt-nodes` runs the shared CPU core without Blender or CUDA. GPU regression commands are documented in [blender/README.md](blender/README.md) and the [node support matrix](docs/shader-nodes.md).

[deslop-please](https://github.com/deterministic-machines/deslop-please) checks the native workspace's architecture. Python bindings live in `vibrt/src/python/`; the renderer and shader compiler share the independent `Progress` interface in `progress.rs`.

## Continuous optimization with Codex

With Codex CLI installed and logged in, start a manual optimization loop:

```powershell
py scripts/auto_optimize.py
```

The loop uses a dedicated worktree, repeats implementation, performance/image
comparisons and relevant tests, then opens a PR per verified improvement. It fixes failed CI on the
same PR and merges after CI passes. Use **Ctrl+C** to interrupt or `py scripts/auto_optimize.py --stop` in
another terminal to finish the current round. See [loop configuration and logs](docs/auto-optimization.md).

Old optimizer build artifacts are cleaned at startup and between rounds, keeping
the newest three runs and recovery/publication checkpoints. Reports and source
snapshots remain available; new experiments wait when free space falls below 10 GiB.
