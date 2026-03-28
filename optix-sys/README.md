# optix-sys

Raw FFI bindings to the [NVIDIA OptiX 9.0.0](https://developer.nvidia.com/rtx/ray-tracing/optix) ray tracing SDK, generated with [bindgen](https://github.com/rust-lang/rust-bindgen).

## What's included

- All OptiX types, enums, and structs from `optix_types.h`
- The `OptixFunctionTable` with all 40+ function pointers from `optix_function_table.h`
- Dynamic library loading (`optix_init`) that mirrors the C `optix_stubs.h` behavior
- Minimal CUDA Driver API bindings (`cuda` module) for device memory management
- Constants for alignment requirements, SBT record sizes, ABI version, etc.

## Requirements

- **NVIDIA OptiX SDK 9.0.0** installed at the default location, or set `OPTIX_ROOT` env var
  - Windows: `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0`
  - Linux: `/usr/local/NVIDIA-OptiX-SDK-9.0.0`
- **LLVM/Clang** installed (for bindgen). Set `LIBCLANG_PATH` if not auto-detected
- **NVIDIA GPU** with driver supporting OptiX 9

## Usage

```toml
[dependencies]
optix-sys = { path = "optix-sys" }
```

```rust
use optix_sys::*;

// Initialize OptiX (loads nvoptix.dll / libnvoptix.so.1)
let table = optix_init().expect("Failed to init OptiX");

// Call functions through the table
unsafe {
    let mut ctx: OptixDeviceContext = std::ptr::null_mut();
    let options = OptixDeviceContextOptions::default();
    (table.optixDeviceContextCreate.unwrap())(cuda_ctx, &options, &mut ctx);
}
```

## Running the example

The example renders a barycentric-colored triangle to a PPM file.

First, compile the CUDA device code to PTX:

```bash
nvcc -ptx examples/devicecode.cu -o examples/devicecode.ptx \
     -I"C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include" \
     -Iexamples --use_fast_math -arch=compute_75
```

Then run:

```bash
cargo run --example simple_render -p optix-sys
```

## Architecture

OptiX is a header-only SDK with no link library. The runtime (`nvoptix.dll` / `libnvoptix.so.1`) is loaded dynamically. The `optix_init()` function:

1. Loads the OptiX shared library
2. Looks up `optixQueryFunctionTable`
3. Populates an `OptixFunctionTable` with all API function pointers

On Windows, if the library isn't found in the system directory, the loader falls back to scanning GPU driver registry entries (matching the C SDK's `optix_stubs.h` behavior).
