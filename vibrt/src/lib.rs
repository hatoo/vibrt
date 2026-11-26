//! In-process Rust API for the vibrt renderer.
//!
//! The Blender addon loads this library via PyO3 (under the `python` feature)
//! and hands scene buffers to Rust in-memory. Rust callers can also use the
//! public scene-loading and rendering APIs directly.
//! There is no standalone CLI binary — the disk-loading path was removed
//! along with the addon's subprocess fallback.

#![allow(clippy::missing_transmute_annotations)]

pub mod camera;
pub mod device;
// Preserve the existing public module paths.
pub use device::{pipeline, shaders as shader_compile};
pub mod gpu_types;
pub mod material;
#[cfg(any(feature = "python", test))]
mod pointiness;
pub mod principled;
mod progress;
mod query_bvh;
pub mod render;
pub mod scene_format;
pub mod scene_loader;
pub use vibrt_nodes::sky;
pub mod transform;

#[cfg(feature = "python")]
mod python;

pub use progress::{Progress, StdoutProgress};
pub use render::{render_to_pixels, RenderOptions, RenderOutput};
pub use scene_loader::{load_scene_from_bytes, LoadedScene};

/// Adapter that turns a `cudarc::driver::DriverError` into our `anyhow::Error`.
/// Sits at the crate root so `principled.rs` and `render.rs` can both `use crate::CudaResultExt`.
pub trait CudaResultExt<T> {
    fn cuda(self) -> anyhow::Result<T>;
}

impl<T> CudaResultExt<T> for std::result::Result<T, cudarc::driver::DriverError> {
    fn cuda(self) -> anyhow::Result<T> {
        self.map_err(|e| anyhow::anyhow!("CUDA error: {e:?}"))
    }
}
