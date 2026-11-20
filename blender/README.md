# vibrt Blender Addon

A Blender render engine that runs vibrt in-process via the bundled
`vibrt_native.pyd` PyO3 extension. There is no subprocess fallback —
the addon won't render unless the native extension is bundled.

## Installation

The simplest path on a fresh checkout:

```
make dev-install
```

This builds `vibrt_native.pyd` (cargo with the `python` feature), drops
it next to the addon's Python sources, and creates a junction/symlink
from `blender/vibrt_blender/` into Blender's user addons dir. Edits to
the Python sources are picked up after a Blender restart without
rezipping.

To produce a redistributable zip with the native extension bundled in:

```
make addon-with-native
```

Then install the resulting `blender/vibrt_blender.zip` via Blender →
`Edit` → `Preferences` → `Add-ons` → `Install...`.

`make addon` (no `-with-native`) packages only the Python sources. The
addon will refuse to render in that configuration — it's only useful
for shipping the source layout.

## Usage

1. In the Render Properties panel, set **Render Engine** to `vibrt`.

2. Add geometry, lights (Point, Sun, Spot, or Area), and a camera.

3. Assign materials using supported shader nodes, including **Principled BSDF**
   and **Mix/Add Shader**. Consult the [support matrix](../docs/shader-nodes.md)
   for node-specific limitations.

4. Optionally set a World background with an **Environment Texture** (HDRI)
   or **Sky Texture**.

5. In the Sampling panel, set **Samples** (`vibrt_spp`) and **Clamp Indirect** (`vibrt_clamp_indirect`).

6. Press `F12` (or `Render > Render Image`). The addon hands the
   evaluated scene directly to `vibrt_native.render(...)` in the same
   process and writes color, depth and named AOV buffers into Blender's
   render passes.

The first completed sample is displayed immediately. Further samples
accumulate on the GPU, and Render Result and the sample counter update up to
10 times per second. Fast samples run in short batches, and only the latest
completed image is transferred for display. The GPU can continue sampling while
Blender displays a snapshot. Batches target at most 50 ms of measured GPU work
(up to 64 one-sample launches); a single slow sample can take longer.
The final sample is always displayed. Esc stops between batches and keeps the
latest preview; optional denoising runs once at the end.
Scene export, compilation, and GPU setup still finish before sampling starts.

On Blender 5.1 with the default Linear Rec.709 working space, procedural skies
are evaluated directly by `vibrt_native`. Preetham, Hosek/Wilkie, single
scattering and multiple scattering use Blender's atmosphere equations and
tables. RGB Curves, Hue/Saturation, Gamma, Invert and supported color/shader
mixes also work, including classroom's daylight-portal sky. Preparation uses
the current unsaved settings and creates no temporary Blender scenes.

Unsupported node graphs, custom OCIO configurations and other Blender versions
use an isolated Cycles worker. Only the world/material inputs are snapshotted;
the worker runs alongside geometry export. Esc cancels the worker and render.
Set `VIBRT_CYCLES_SKY=1` before launching Blender to force this reference path.
Building the native extension now also requires a C++17 compiler; bundled
model sources and license notices are described in
[`vibrt-nodes/third_party/sky`](../vibrt-nodes/third_party/sky/README.md).

Sky and shader-node calculations live in the independent
[`vibrt-nodes`](../vibrt-nodes/README.md) crate. Python reads Blender nodes and
tables, while native CPU batches and the CUDA graph interpreter share the
node implementations. `cargo test -p vibrt-nodes` tests this core without
Blender or CUDA. The Python binding tests require a staged native extension.

For regression checks, run `make test` with a POSIX shell, then:

```
blender -b --factory-startup --python-exit-code 1 --python scripts/test_native_sky.py
blender -b --factory-startup --python-exit-code 1 --python scripts/test_sky_bake.py
blender -b --factory-startup --python-exit-code 1 --python scripts/test_progressive_render.py
blender scene.blend --python scripts/test_render_job.py -- --expect-sky --expect-progress
```

The last command tests two asynchronous F12-style renders and exits Blender;
use a separate process. It writes previews under `target/render-job-test/`
without saving the scene or preferences. Omit `--expect-sky` for scenes without
procedural sky or portal lighting. Add `--expect-native-sky` to assert that a
supported sky never starts a Cycles worker. The native-sky test compares sky
models, sun discs, color processing and camera-ray splits against Cycles.

The progressive GPU test compares preview-enabled and preview-disabled renders,
normalization of the first sample, final denoising, batched accumulation,
snapshot lifetime, named AOVs, cancellation, and callback errors. The render-job
test inspects actual Blender passes while F12 is still running.

Both modes queue one-sample launches without waiting for each sample on the CPU.
Omitting a preview callback avoids intermediate pixel readback. The native
`preview_interval_ms=0` option explicitly requests every-sample display and waits
after each sample when a preview callback is supplied.

## Supported features

The [shader node support matrix](../docs/shader-nodes.md) inventories all 99
Blender 5.1.1 node types, including partial, legacy and unsupported paths.
The general graph evaluates linked material parameters, procedural/image
textures, color/vector math and Mix/Add shaders on the GPU. It is limited to
256 value nodes per graph, 16 closure leaves and 8 shader-tree levels.
Cycles-compatible node math does not imply identical BSDF or volume models.

### Scene synchronization

Blender 5.1 uses a native mesh reader modeled on Cycles' synchronization:
Rust reads Blender's evaluated position, corner, triangle, normal, UV, and
float-color arrays in place, then writes the final renderer buffers once.
RNA collection pointers locate these primitive arrays; no hardcoded C++
`Mesh` offsets or custom Blender build are required. Byte colors still use
Blender's color-managed getter. All references are consumed synchronously
before `to_mesh_clear()`; only owned renderer arrays survive the call.

The scene description crosses PyO3 as a live dictionary, without JSON
encoding/decoding. Rust also processes color conversion and bilinear resizing
on borrowed image arrays. Material graph translation and image pixel retrieval
still use Blender's Python API; this is not a complete port of Cycles' shader
synchronization. GPU uploads remain necessary.

CPU texture buffers stay float32 RGBA. Before upload, redundant channels are
removed losslessly to use L, LA, RGB or RGBA on the GPU; image dimensions and
float32 values are preserved. This is separate from optional texture resizing.

Direct mesh reads are enabled only for Blender 5.1, whose RNA layouts are
verified against Blender's source. Other versions use `foreach_get`.
Set `VIBRT_PYTHON_EXPORT=1` before launching Blender to compare the original
mesh/pixel/JSON path. The default uses native synchronization.

```
blender -b --factory-startup --python-exit-code 1 --python scripts/test_native_scene.py
blender -b scene.blend --python-exit-code 1 --python scripts/test_native_scene.py -- --scene
```

These check exact geometry/normal/UV/color parity, evaluated-mesh lifetime,
pixel-operation accuracy, and scene schema errors. Run the asynchronous
render-job check above after modifying the memory reader.

### Rendering

- Camera: perspective, lens shift, and circular-aperture depth of field.
- Meshes: triangulated, per-corner normals and UVs, multi-material slots (per-triangle material index).
- Materials: linked Principled inputs, image/procedural textures, normal/bump mapping, displacement and Mix/Add closure trees. Supported graphs run on the GPU; legacy conversion may bake textures or approximate unsupported inputs. Standalone Chiang Hair and Burley Subsurface Scattering have dedicated closures; Principled SSS remains approximate.
- Lights: Point, Sun, Spot, Area (square/rectangle), IES profiles, and emissive-mesh NEE.
- World: constant background, importance-sampled Environment Texture, or supported native Sky Texture graphs; other world graphs use the isolated Cycles preparation path.
- Volumes: homogeneous Principled / Absorption / Scatter (mesh-bounded and world-volume).
- Transparent shadows through transmissive / alpha-cutout surfaces.
- Output: linear Combined, depth/Mist and named Color/Value AOV passes; Blender manages display and file output.

## Not supported yet

- Viewport IPR (live preview).
- Motion blur, heterogeneous volumes (OpenVDB), and modern Curves geometry.
- Thin film, Huang Hair and random-walk SSS. See the support matrix for other partial node modes.
