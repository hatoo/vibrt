//! In-process Rust API for the vibrt renderer.
//!
//! The binary (`src/main.rs`) is a thin CLI wrapper that calls into this
//! library. The Blender addon links the same library directly via PyO3 (under
//! the `python` feature) so it can hand scene buffers to Rust without going
//! through the disk roundtrip.

#![allow(clippy::missing_transmute_annotations)]

pub mod camera;
pub mod color_fold;
pub mod gpu_types;
pub mod image_io;
pub mod pipeline;
pub mod principled;
pub mod render;
pub mod scene_format;
pub mod scene_loader;
pub mod transform;

#[cfg(feature = "python")]
mod python;

pub use render::{render_to_pixels, Progress, RenderOptions, RenderOutput, StdoutProgress};
pub use scene_loader::{load_scene_from_bytes, load_scene_from_path, LoadedScene};

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
