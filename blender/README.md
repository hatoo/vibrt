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

3. Assign materials using the **Principled BSDF** shader.

4. Optionally set a World background with an **Environment Texture** (HDRI).

5. In the Sampling panel, set **Samples** (`vibrt_spp`) and **Clamp Indirect** (`vibrt_clamp_indirect`).

6. Press `F12` (or `Render > Render Image`). The addon hands the
   evaluated scene directly to `vibrt_native.render(...)` — no temp
   files, no subprocess — and copies the resulting RGBA float buffer
   into Blender's Combined pass.

## Supported features

- Camera: perspective only (no DoF yet).
- Meshes: triangulated, per-corner normals and UVs, multi-material slots (per-triangle material index).
- Materials: Principled BSDF — base color, metallic, roughness, IOR, transmission, emission, anisotropy (+ rotation), coat (weight / roughness / IOR), sheen (weight / roughness / tint), subsurface (weight / radius / anisotropy), alpha cutout, normal map (with strength), bump, displacement. Image textures on base color / normal / roughness / metallic. Colour- and scalar-math node chains between TexImage and the BSDF are traversed; the effect of RGBCurve, Gamma, BrightContrast, Invert, HueSaturation, ColorRamp, Clamp and Mix/MixRGB (MIX / MULTIPLY / ADD / SUBTRACT, when the non-texture side is constant) is baked into the exported texture pixels. Other node effects are approximated by pass-through.
- Lights: Point, Sun, Spot, Area (square/rectangle).
- World: constant colour background or Environment Texture (importance-sampled).
- Volumes: homogeneous Principled / Absorption / Scatter (mesh-bounded and world-volume).
- Transparent shadows through transmissive / alpha-cutout surfaces.
- Output: linear float image loaded into Blender's image editor.

## Not supported yet

- Viewport IPR (live preview).
- Depth of field, motion blur, heterogeneous volumes (OpenVDB).
- Thin-film, true SSS (subsurface is currently diffuse-blend approximation).
