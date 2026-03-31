# vivrt

A Rust workspace for GPU ray tracing with NVIDIA OptiX 9.

## Crates

| Crate | Description |
|-------|-------------|
| [`optix-sys`](optix-sys/) | Raw FFI bindings to OptiX 9.0.0 (bindgen) |
| [`optix`](optix/) | Safe Rust wrapper with RAII, builders, and type-safe enums |
| [`pbrt-parser`](pbrt-parser/) | Zero-dependency parser for PBRTv4 scene files |
| [`vivrt`](vivrt/) | OptiX path tracing renderer for PBRTv4 scenes |

## vivrt renderer

A GPU path tracer that reads [PBRTv4](https://pbrt.org/fileformat-v4) scene files and renders them using OptiX hardware ray tracing.

### Features

- Path tracing with configurable depth and samples per pixel
- Materials: diffuse, coated diffuse (GGX), conductor (metallic Fresnel), dielectric (glass)
- Geometry: triangle meshes, PLY meshes (binary, gzip), spheres (built-in intersection), bilinear patches, Loop subdivision
- Lighting: distant, infinite, sphere area lights, triangle area lights with next-event estimation
- Imagemap textures with bilinear filtering, checkerboard procedural texture
- CUDA device code compiled at runtime via NVRTC

### Usage

```bash
cargo run --release -p vivrt -- scene.pbrt
cargo run --release -p vivrt -- scene.pbrt --spp 64 --width 800 --height 600
```

### Example renders

```bash
# Simple glass sphere on checkerboard
cargo run --release -p vivrt -- test.pbrt

# Killeroo scene (coated diffuse + area lights)
cargo run --release -p vivrt -- path/to/killeroos/killeroo-simple.pbrt --spp 64

# Crown scene (conductors, dielectrics, textures, 793 objects)
cargo run --release -p vivrt -- path/to/crown/crown.pbrt --spp 32 --width 500 --height 700
```

## Requirements

- **NVIDIA OptiX SDK 9.0.0** -- set `OPTIX_ROOT` or install at default location
- **CUDA Toolkit** -- for NVRTC runtime compilation
- **LLVM/Clang** -- for bindgen (set `LIBCLANG_PATH` if not auto-detected)
- **NVIDIA GPU** with driver supporting OptiX 9

## Building

```bash
cargo build --release -p vivrt
```
