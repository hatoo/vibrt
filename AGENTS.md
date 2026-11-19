# Repository Guidelines

## Project Structure & Module Organization

vibrt renders inside Blender through the PyO3 `vibrt_native` extension.

- `vibrt/src/`: Rust renderer, scene loading, materials, and CUDA kernels (`devicecode.cu`/`devicecode.h`).
- `optix/`: safe Rust OptiX wrappers; `optix-sys/`: generated FFI bindings and SDK setup.
- `blender/vibrt_blender/`: addon, scene/material exporters, and adjacent Python tests.
- `scripts/`: headless rendering, Cycles comparisons, and layout checks.
- `test_scenes/`: scenes and IES fixtures. Most large scene directories are gitignored; `ies_light` is tracked.

## Build, Test, and Development Commands

Run commands from the repository root. Native development requires Rust, Python 3.11+, OptiX SDK 9.0.0, CUDA/NVRTC, and LLVM/Clang. Rendering also requires Blender and a compatible NVIDIA GPU. Make's test loop requires a POSIX shell; override its Python launcher with `PYTHON=python` if needed.

- `cargo build --release -p vibrt --features python`: build the native extension.
- `make native-build`: build and stage the extension beside the addon.
- `make dev-install`: build, stage, and link the addon into Blender's user directory.
- `make addon-with-native`: package a usable addon ZIP.
- `make ies_light-preview SPP=32 PCT=50`: render the tracked IES scene.

Set `OPTIX_ROOT` for a nondefault SDK location. Check `.cargo/config.toml`, which forces a Windows `LIBCLANG_PATH`.

## Coding Style & Naming Conventions

Use four-space indentation in Rust/Python and the existing two-space CUDA style. Use `snake_case` for functions/modules, `PascalCase` for types/classes, and `UPPER_SNAKE_CASE` for constants. Check Rust with `cargo fmt --all -- --check` and `cargo clippy --workspace --features python`. No custom formatter configuration is checked in.

Keep `gpu_types.rs` and `devicecode.h` layouts synchronized. Update exporters and Rust scene schemas together; preserve borrowed buffer lifetimes across PyO3.

## Testing Guidelines

Run `make test` for struct-layout checks and Python `unittest` suites. Python tests need NumPy, but run without Blender/CUDA. Name files `test_*.py` and methods `test_*`; individual suites run with `py blender/vibrt_blender/test_bin_writer.py`. No coverage threshold is configured.

For rendering changes, compare the preview with `make ies_light-cycles SPP=32 PCT=50`, then exercise an affected scene.

## Commit & Pull Request Guidelines

Use concise, action-focused subjects, often prefixed by the component, as in `exporter: flip IES frame`. PRs should explain the problem, resulting behavior, validation commands, and relevant issues. Include comparison renders for visual changes. Keep generated extensions, ZIPs, renders, and large local scenes out of commits.
