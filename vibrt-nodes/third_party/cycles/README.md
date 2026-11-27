# Cycles source provenance

The sources live in the official Blender git submodule at
[`../blender`](../blender), pinned to Blender **v5.1.1**, commit
`b70da489d7f4eae7760cc0a50cc6c60cd5464205`.
The former copies and SHA256 manifest have been removed; the gitlink identifies
the exact source tree. No files inside the submodule are patched.

Initialize it before building from a fresh checkout:

```sh
git submodule update --init --depth 1
```

Only `intern/cycles` is needed. A local sparse checkout can reduce the working
tree with `git -C vibrt-nodes/third_party/blender sparse-checkout set intern/cycles`.
Blender itself does not need to be built. Installed addons embed the generated
code/tables and do not read this checkout at runtime.

`vibrt-nodes/build.rs` extracts numerical helpers into an OUT_DIR header, removing
Blender includes, namespace macros and SVM bytecode entry points. It reads:

- `kernel/svm/`: enums, Math/Mix, procedural textures, radial tiling, Mapping,
  scalar/vector Map Range and Vector Rotate. Fresnel uses the closure helper.
- `util/hash.h`, `util/color.h`, `util/math_fast.h`: hashes, color conversion and
  inverse error function.
- `kernel/closure/`: Diffuse/Oren-Nayar, Sheen/Velvet, microfacet/Fresnel, Chiang
  hair, legacy Hair reflection/transmission, Toon, Burley/dipole profiles and
  volume phase functions.
- `kernel/sample/mapping.h`, `kernel/util/lookup_table.h`, `kernel/tables.h`
  and `scene/shader.tables`: sampling, interpolation and original numerical tables.
- `kernel/integrator/subsurface_random_walk.h`: coefficient fit, Dwivedi phase,
  diffusion length and spectral distance PDF. The renderer combines classic,
  forward-guided and backward-guided walks with RGB balance weights.
- `util/ies.h`, `util/ies.cpp`, `kernel/util/ies.h`: Type A/B/C parsing, symmetry,
  radiometric conversion and cubic interpolation. Rust validates sizes, angles
  and intensities before entering the C++ parser. Legacy scene tables are
  expanded as Type C; processed profiles carry an explicit schema flag.
- `kernel/sample/sobol_burley.h` and `kernel/sample/util.h`: Owen-scrambled Sobol,
  using independent streams per pixel and path event.
- `kernel/light/triangle.h`, `util/math_intersect.h`: triangle solid-angle/area
  sampling and matching hit PDFs. The adapter supplies world-space vertices.
- `kernel/light/tree.h`: subtended-angle bounds, importance and branch selection
  probabilities. A renderer-owned median-split tree selects rectangular lights
  and mesh-light groups. Triangles within a group retain the power CDF; explicit
  point/spot/sun and environment strategies retain their existing selection.
- `kernel/bvh/util.h`: scale-aware ray offsets for surface and shadow origins.

`native/cycles_compat.h` and `native/closures_compat.h` provide platform types
and the small numerical context. `native/closure_tables.h` maps table addresses;
thin-film CMF data is converted to scene-linear Rec.709 as in
`ShaderManager::compute_thin_film_table`.
`native/bsdf.h` adapts layer assembly from `kernel/svm/closure.h` to explicit
parameters. Setup, evaluation, sampling, Fresnel and PDFs call upstream functions.
`native/core.h` maps legacy wire-format IDs and edge conventions to the same
Math/Mix/color helpers used by `native/shader.h`.
`native/reuse_compat.h`, `triangle_compat.h` and `light_tree_compat.h` adapt
stack access, world-space geometry and plain tree nodes without Blender scene
ownership. `hair_adapter.h` re-evaluates sampled directions to correct the
pinned legacy reflection sampler's reported PDF (its sampled azimuth and
evaluation use different roughness scaling). No upstream file is changed.

Image, normal, tangent and scene-query adapters follow the corresponding SVM
sources. Toon, AO, Bevel, Wireframe, Raycast and subsurface traversal use vibrt's
geometry; Blender scene management and OSL are not linked.

Most selected code is Apache-2.0, copyright Blender Foundation/Blender Authors.
Noise, legacy Hair and Toon include BSD-3-Clause code from Sony Pictures Imageworks and
Blender Foundation. Parts of Voronoi are MIT, copyright Inigo Quilez.
Individual SPDX headers are authoritative. License texts and attribution are
bundled in [`blender/vibrt_blender/licenses`](../../../blender/vibrt_blender/licenses).
The Blender repository also contains code outside these selected sources;
those files are not compiled into vibrt by this integration.
