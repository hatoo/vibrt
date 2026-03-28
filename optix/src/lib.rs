pub mod error;
mod sys;
pub mod types;
pub mod context;
pub mod module;
pub mod program_group;
pub mod pipeline;
pub mod accel;
pub mod sbt;
pub mod denoiser;

use crate::error::{OptixError, Result};
use crate::sys::FunctionTable;
use std::sync::Arc;

// Re-export commonly used types at the crate root
pub use context::{DeviceContext, DeviceContextOptions};
pub use error::WithLog;
pub use module::{Module, ModuleCompileOptions};
pub use pipeline::{Pipeline, PipelineCompileOptions, PipelineLinkOptions};
pub use program_group::ProgramGroup;
pub use sbt::{SbtRecord, SbtRecordHeader, ShaderBindingTableBuilder};
pub use types::*;

// Re-export optix-sys types that appear in the public API
pub use optix_sys::{CUcontext, CUdeviceptr, CUstream, OptixTraversableHandle};
pub use optix_sys::OPTIX_SBT_RECORD_HEADER_SIZE;

/// Handle to an initialized OptiX instance.
///
/// Created by calling [`init()`]. The function table is shared with all
/// resources created from this handle via `Arc`.
pub struct Optix {
    pub(crate) table: Arc<FunctionTable>,
}

/// Initialize OptiX and return an `Optix` handle.
///
/// This dynamically loads the OptiX library and populates the function table.
/// Call this once at startup before creating any OptiX resources.
pub fn init() -> Result<Optix> {
    let raw_table = optix_sys::optix_init().map_err(OptixError::from_raw)?;
    Ok(Optix {
        table: FunctionTable::new(raw_table),
    })
}
