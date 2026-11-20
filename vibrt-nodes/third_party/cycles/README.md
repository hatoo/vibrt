# Cycles shader math

These headers are unmodified copies from Blender **v5.1.1**, commit
`b70da489d7f4eae7760cc0a50cc6c60cd5464205`.
`SHA256.json` records the exact bytes. Upstream paths are:

- `intern/cycles/kernel/svm/`: `types.h`, `math_util.h`, `color_util.h`,
  `fractal_noise.h`, `noisetex.h`, `voronoi.h`, `wave.h`, `brick.h`,
  `checker.h`, `gradient.h`, `magic.h`, `gabor.h`, `mapping.h`,
  `map_range.h`, `blackbody.h`, `wavelength.h`, `vector_rotate.h`, `noise.h`,
  `mapping_util.h`, `radial_tiling_shared.h`, `ao.h`, `bevel.h`, `raycast.h`,
  `wireframe.h`.
- `intern/cycles/util/`: `hash.h`.
- `intern/cycles/kernel/`: `tables.h`.
- `intern/cycles/kernel/closure/`: `bsdf_principled_hair_chiang.h`,
  `bsdf_toon.h`, `bsdf_ray_portal.h`, `bssrdf.h`, `volume_util.h`.
- `intern/cycles/kernel/integrator/`: `subsurface_disk.h`.

Verify these locations against the upstream link and manifest when updating:
[Blender source](https://github.com/blender/blender/tree/v5.1.1).

`build.rs` extracts the numerical helpers and enums into an OUT_DIR header,
removing Blender includes, namespace macros and SVM bytecode entry points.
`native/cycles_compat.h` supplies vector/math types without Blender dependencies.
`native/shader.h` adapts graph inputs, image sampling, coordinate transforms,
normal maps, Fresnel and curve tables to vibrt's shading context. Image, normal,
Fresnel and tangent adapters also follow the corresponding v5.1.1 SVM sources.
These adapters are adaptations; the files in this directory remain unchanged.
The build also extracts the unmodified Chiang hair closure, Burley radial
profiles and volume phase functions. `native/closures_compat.h` supplies their
small numerical context. Toon, AO, Bevel, Wireframe, Raycast and subsurface disk
sampling adapt the referenced algorithms to vibrt's geometry and traversal;
they do not import Blender's scene objects or traversal implementation.

Most code is Apache-2.0, copyright Blender Foundation/Blender Authors.
`noise.h` and `bsdf_toon.h` are BSD-3-Clause,
copyright Sony Pictures Imageworks (2009–2010)
and Blender Foundation (2011–2022). Parts of `voronoi.h` are MIT,
copyright Inigo Quilez (2013). Individual SPDX headers are authoritative.
License texts and attribution are bundled in
[`blender/vibrt_blender/licenses`](../../../blender/vibrt_blender/licenses).

This is a shared CPU/CUDA node evaluator with selected numerical closures,
not a linked Cycles renderer. It does not include Cycles scene management,
OSL, point-cloud or volume-grid support.
