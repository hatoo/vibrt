# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build / run

Toolchain prerequisites: Rust, a C++17 compiler, NVIDIA OptiX SDK 9.0.0 (`OPTIX_ROOT`), CUDA Toolkit (NVRTC), LLVM/Clang (`LIBCLANG_PATH` — forced in `.cargo/config.toml` to `C:\Program Files\LLVM\bin`), Python 3.11+ (PyO3 abi3-py311), and an OptiX-capable NVIDIA GPU for rendering.

There is **no standalone CLI binary**. The renderer ships only as a PyO3 extension (`vibrt_native.pyd`) that the Blender addon loads in-process. The disk-loading path was removed alongside the addon's subprocess fallback.

```bash
cargo build --release -p vibrt --features python    # produces target/release/vibrt_native.dll (the cdylib)
cargo build --release -p vibrt                      # libraries without the Python bindings
make native-build                                   # build and stage the importable extension beside the addon
make dev-install                                    # build .pyd, stage next to addon source, junction into Blender's user addons
```

CPU tests run without Blender or GPU rendering. Python suites need NumPy; native binding tests require a staged extension. `make test` uses a POSIX shell:

```bash
make test                       # layout checks, Rust node tests, addon test_*.py
cargo test -p vibrt-nodes        # shared C++/Rust node core, no CUDA/OptiX needed
cargo fmt --all -- --check
cargo clippy --workspace --features python
```

End-to-end verification uses Blender scripts and comparison renders. The tracked IES scene is available on a fresh checkout:

```bash
make junk_shop-preview      # vibrt render of test_scenes/junk_shop/junk_shop/junk_shop.blend
make junk_shop-cycles       # Cycles reference render of the same .blend
make ies_light-preview SPP=32 PCT=50
make ies_light-cycles SPP=32 PCT=50
make previews              # all scenes listed in Makefile's BLEND_SCENES
make cycles-previews       # matching Cycles references
```

`make` overrides: `PYTHON=`, `SPP=`, `PCT=` (render resolution percentage), `DENOISE=1`. Large .blend scenes such as junk_shop and classroom are local, gitignored assets; see the Makefile for their exact paths. GPU node, sky and progressive-render checks are documented in `blender/README.md` and `docs/shader-nodes.md`.

The preview pipeline spawns Blender headless (`scripts/render_blend.py` → `scripts/_blender_render.py`), force-loads the working-tree addon, and triggers a single F12 render via `vibrt_native.render(...)`. There is no `vibrt.exe` and no scene.json/scene.bin on disk.

## Architecture

This workspace has four Rust crates and a Python Blender addon. The render data flow is **Blender → (scene dict, mesh_blobs[], texture_arrays[]) → vibrt_native.render → color/depth/AOV buffers → Blender render passes**. PyO3 pins Python-owned arrays for the duration of rendering. JSON text remains an alternative to the live scene dict. Unsupported sky/world preparation can run in an isolated Cycles worker; the vibrt render itself stays in-process.

### Rust crates

- **`optix-sys/`** — bindgen FFI to OptiX 9.0.0. Pure unsafe.
- **`optix/`** — safe wrapper: RAII handles, builders, typed enums, error/log surfacing, SBT and denoiser helpers. The renderer also uses `optix-sys` POD types and device addresses where the wrapper requires them.
- **`vibrt-nodes/`** — renderer-independent shader graph schemas, validation and shared C++ CPU/CUDA node math, plus CPU sky models. Constant folding uses the same color implementation. See `docs/shader-nodes.md` for support and limits.
- **`vibrt/`** — the renderer.
  - `[lib] name = "vibrt_native"`; `crate-type = ["rlib", "cdylib"]`. The cdylib lands at `target/release/vibrt_native.dll` and is staged as `blender/vibrt_blender/vibrt_native.pyd` so Blender imports it as `vibrt_blender.vibrt_native`.
  - The `python` feature pulls in `pyo3` + `numpy` and enables the importable module. Both library types are built even without that feature.
  - There is no `[[bin]]` section. Blender uses the PyO3 module; Rust callers can use the public library API.

### `vibrt/src/` map

- **`scene_format.rs`** — versioned (`version: 1`) scene schema. Mesh fields are plain `BlobIdx` indices into separate byte buffers; `TextureDesc.array_index` selects a float32 texture array. Keep exporter, schema and Python dict deserializer synchronized.
- **`scene_loader.rs`** — `LoadedScene<'bin>` borrows mesh blobs and linear RGBA texture arrays through `Cow`. sRGB conversion, RGB expansion and geometric displacement allocate owned buffers when needed. Preserve borrowed lifetimes and avoid unnecessary copies.
  - `load_scene(file, mesh_blobs, texture_arrays)` handles an already-deserialized scene; `load_scene_from_bytes(json_text, mesh_blobs, texture_arrays)` parses JSON first.
- **`render.rs`** — context creation, uploads, acceleration structures, SBT, one-sample launches, optional denoise and readback. `RenderOutput` contains `pixels`, `depth`, named `aovs`, `width` and `height`.
  - `Progress` handles logging, cancellation and previews. `PyProgress` forwards these to the addon's Python callbacks; `StdoutProgress` logs without previews. Short batches queue one-sample launches between CPU waits, targeting 50 ms of measured GPU time with a 64-sample cap. First/final snapshots are always displayed; intervening previews default to a 100 ms cadence. The next batch can run during display; cancellation/errors drain pending work before GPU resources are released.
- **`pipeline.rs` / `shader_compile.rs`** — scene-specific NVRTC/OptiX variants shared by rendering and shader precompilation. Headers are embedded and PTX is cached on disk. GGX energy tables are generated by `vibrt/build.rs` and embedded; `build_envmap_cdf()` builds HDRI sampling tables.
- **Volumes** are homogeneous-only. Per-material and world `VolumeParams` become `VolumeGpu` coefficients with a selected phase function (HG or supported extended phases). `volume_only` hides the boundary surface. The fixed-depth `VolumeStack` holds up to four mesh volumes, with world volume implicit. Heterogeneous grids and spatial density graphs remain unsupported.
- **`devicecode.cu` / `devicecode.h`** — embedded OptiX device programs, compiled by NVRTC. Rebuild and stage the extension after native edits; restart Blender to load the new binary. Opaque instances can skip any-hit; alpha, transmission and volume cases retain it.
- **`principled.rs`** — uploads textures, legacy color graphs, typed shader graphs, closure trees and volume coefficients. GPU textures use lossless float32 L/LA/RGB/RGBA layouts selected from their RGBA source.
- **`gpu_types.rs`** — host/device POD layouts, paired with `devicecode.h`. Shared node layouts live in `vibrt-nodes`; `scripts/check_struct_layouts.py` checks both.
- **`python.rs`** — `render(scene_json, mesh_blobs, opts, log_cb=None, cancel_cb=None, texture_arrays=None, preview_cb=None, aov_outputs=None)` returns `(pixels, depth)`. Pixels are `(h, w, 4)` float32; depth is `(h, w)`. Named AOVs fill the optional dict. Buffer views pin their Python sources while `py.allow_threads` releases the GIL for rendering; callbacks briefly reacquire it.
- **`python_scene.rs`, `blender_mesh.rs`, `python_pixels.rs`, `python_nodes.rs`** — live-dict deserialization, evaluated mesh reads, image operations and shared-node bindings. Blender RNA pointers are consumed before `to_mesh_clear()`; only owned renderer arrays survive the native mesh-read call.

### Python addon (`blender/vibrt_blender/`)

- **`engine.py`** — `VibrtRenderEngine(bpy.types.RenderEngine)`. F12 entry point. The addon renders **only** through the bundled `vibrt_native.pyd`; if the extension isn't importable the engine reports an error and stops. There is no subprocess fallback.
- **`exporter.py` / `bin_writer.py`** — `export_scene_to_memory(..., serialize=False)` returns a live dict and two lists of contiguous NumPy arrays. Mesh blobs are indexed directly; texture buffers travel separately as f32 views. Blender 5.1 uses native evaluated-mesh reads, with `foreach_get` fallback on other versions or `VIBRT_PYTHON_EXPORT=1`.
- **`material_export.py` / `shader_graph.py`** — typed GPU graphs and Mix/Add closure trees, with legacy conversion/baking fallbacks. The general graph supports up to 256 value nodes, 16 closure leaves and 8 tree levels. Emissive meshes use triangle NEE; selected camera-hidden panels and IES emitters use rect promotion.
- **`color_fold.py` / `node_ops.py`** — call the shared native CPU evaluator for constant folding and image color operations. Without a staged extension, constant folding leaves the graph intact.
- **`hair_export.py`** — legacy particle hair becomes ribbon meshes in bounded NumPy batches, with tangent and Hair Info attributes. `rendered_child_count` is per-parent; children are approximated. Modern Curves geometry is not exported.
- **`native_sky.py` / `sky_bake.py`** — supported Blender 5.1/default Linear Rec.709 sky graphs use native models; other graphs/configurations use an isolated Cycles worker, overlapped with export.
- **`runner.py`** — `find_native_module()` and `run_render_inproc()`. The latter is a thin wrapper over `vibrt_native.render(...)` that forwards `texture_arrays`.
- **`build_addon.py`** — packages `vibrt_blender.zip`. `--with-native` builds the cdylib via cargo and stages it as `vibrt_native.pyd`/`.so`/`.dylib`. `--stage-only` does the build + copy without zipping (used by `make dev-install` for the junctioned-addon workflow). A no-`--with-native` zip won't actually render — the addon errors out without the extension.

## Scene-format invariants worth knowing

- Right-handed, Z-up, metres throughout. No axis conversion is needed. Flatten Blender matrices by row into the schema's row-major arrays.
- Camera looks down local −Z (Blender convention).
- Texture `colorspace` is either `"srgb"` or `"linear"`. Host-side baking or scene loading performs color conversion; the GPU sees linear data. Channel compaction preserves float32 samples and dimensions.
- Mesh material indices are per-triangle (`MeshDesc::material_indices`), indexing into `ObjectDesc::materials` (the per-object slot list), not the global `materials` array directly.
- Area lights: Blender area lights emit along local −Z; vibrt's `area_rect` expects local +Z. The exporter flips the third column of the transform — don't introduce a second flip on either side.

## Memory / cost intuition

- Image retrieval still allocates float32 RGBA on the host (16 bytes per pixel); GPU uploads and optional color conversion/packing also cost time. Measure export, preparation and sampling separately instead of assuming one dominates every scene.
- GPU texture channel compaction drops only redundant RGB channels or alpha equal to one. It does not quantize to fp16/u8 or change resolution. Preserve buffer lifetimes until asynchronous uploads finish.
- PTX is cached in the system temporary directory under `vibrt-ptx-v1`, keyed by source and compile options; OptiX also uses its cache. The Render Properties **Shaders → Precompile Shaders** button warms the required variant, without mesh uploads or lighting preparation. New node/closure requirements may require a different variant. Cancellation is checked around compilation; it cannot interrupt an active NVRTC call.

## Helper scripts (`scripts/`)

- **`render_blend.py`** — top-level entry point used by `make <scene>-preview`. Spawns Blender headless and runs `_blender_render.py` inside it.
- **`_blender_render.py`** — runs inside `blender -b … --python …`. Force-imports the working-tree addon, sets `render.engine = "VIBRT"`, calls `bpy.ops.render.render(write_still=True)`.
- **`benchmark_render.py`** — exports once and renders repeatedly, separating preparation from sample timings. Run inside Blender; usage and comparison caveats are in `docs/shader-nodes.md`. `--batch` disables preview readback/callbacks while retaining per-sample launches.
- **`render_cycles.py` / `_blender_cycles.py`** — Cycles reference renderer (no vibrt involved). Used for side-by-side comparison.
