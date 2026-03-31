# optix

Safe Rust wrapper for the [NVIDIA OptiX 9.0.0](https://developer.nvidia.com/rtx/ray-tracing/optix) ray tracing SDK, built on top of [`optix-sys`](../optix-sys).

## Features

- **RAII resource management** -- `DeviceContext`, `Module`, `ProgramGroup`, `Pipeline`, and `Denoiser` automatically clean up on drop
- **Type-safe enums** -- proper Rust enums instead of raw integer constants
- **Builder patterns** -- `PipelineCompileOptions`, `HitgroupBuilder`, `ShaderBindingTableBuilder`, etc.
- **Error handling** -- `Result<T, OptixError>` with `?` support and 40+ typed error variants
- **Compiler log capture** -- `WithLog<T>` exposes compilation warnings/errors even on success
- **Bitflags** -- `BuildFlags`, `GeometryFlags`, `ExceptionFlags`, etc. via the `bitflags` crate
- **Built-in sphere intersection** -- `Module::builtin_is()` for hardware-accelerated sphere ray tracing

## Requirements

Same as `optix-sys`:
- NVIDIA OptiX SDK 9.0.0
- LLVM/Clang (for bindgen)
- NVIDIA GPU with compatible driver

## Usage

```toml
[dependencies]
optix = { path = "optix" }
cudarc = { version = "0.19", default-features = false, features = ["driver", "dynamic-loading", "cuda-version-from-build-system", "nvrtc"] }
```

### Minimal workflow

```rust
use optix::*;
use optix::accel::{self, AccelBuildOptions, BuildInput, TriangleArrayInput};

// 1. Initialize
let optix = optix::init()?;
let ctx = DeviceContext::new(&optix, cuda_context, &DeviceContextOptions::default())?;

// 2. Compile module from PTX (compiled at runtime via NVRTC)
let pipeline_options = PipelineCompileOptions::new("params")
    .num_payload_values(3)
    .num_attribute_values(3);
let module = Module::new(&ctx, &ModuleCompileOptions::default(), &pipeline_options, ptx)?.value;

// 3. Create program groups
let raygen = ProgramGroup::raygen(&ctx, &module, "__raygen__rg")?.value;
let miss = ProgramGroup::miss(&ctx, &module, "__miss__ms")?.value;
let hitgroup = ProgramGroup::hitgroup(&ctx)
    .closest_hit(&module, "__closesthit__ch")
    .build()?.value;

// 4. Create pipeline
let pipeline = Pipeline::new(
    &ctx, &pipeline_options,
    &PipelineLinkOptions { max_trace_depth: 1 },
    &[&raygen, &miss, &hitgroup],
)?.value;

// 5. Build acceleration structure
let sizes = accel::accel_compute_memory_usage(&ctx, &build_options, &build_inputs)?;
let gas_handle = accel::accel_build(&ctx, stream, &build_options, &build_inputs, ...)?;

// 6. Create SBT records
let sbt_record = SbtRecord::new(&raygen, MyRayGenData {})?;
let sbt = ShaderBindingTableBuilder::new(d_raygen)
    .miss_records(d_miss, stride, 1)
    .hitgroup_records(d_hitgroup, stride, 1)
    .build()?;

// 7. Launch
pipeline.launch(stream, d_params, params_size, &sbt, width, height, 1)?;
```

## Running the example

The example compiles CUDA device code at runtime via NVRTC:

```bash
cargo run --example simple_render -p optix
```

This renders a barycentric-colored triangle on a dark blue background and saves it as `output.ppm`.

## API overview

| Module | Key types |
|---|---|
| `context` | `DeviceContext`, `DeviceContextOptions` |
| `module` | `Module`, `ModuleCompileOptions` |
| `program_group` | `ProgramGroup`, `HitgroupBuilder`, `CallablesBuilder` |
| `pipeline` | `Pipeline`, `PipelineCompileOptions`, `PipelineLinkOptions` |
| `accel` | `BuildInput`, `TriangleArrayInput`, `SphereArrayInput`, `accel_build()`, `accel_compact()` |
| `sbt` | `SbtRecord<T>`, `SbtRecordHeader`, `ShaderBindingTableBuilder` |
| `denoiser` | `Denoiser`, `DenoiserOptions`, `Image2D` |
| `types` | `BuildFlags`, `GeometryFlags`, `VertexFormat`, `PrimitiveType`, ... |
| `error` | `OptixError`, `Result<T>`, `WithLog<T>` |

## Design notes

- **No CUDA memory ownership** -- the wrapper never allocates or frees GPU memory. Use [cudarc](https://crates.io/crates/cudarc) for CUDA memory management.
- **No lifetime parameters on resource types** -- `Module`, `Pipeline`, etc. hold an `Arc<FunctionTable>` internally. This lets you store them freely in structs without infectious lifetimes.
- **Natural drop ordering** -- declare the CUDA context before OptiX resources and Rust's reverse drop order handles cleanup automatically.
