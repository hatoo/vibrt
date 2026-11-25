//! Acceleration-structure operations tied to a borrowed device context.

use crate::error::{self, Result};
use crate::sys::FunctionTable;
use std::ptr;

pub(crate) struct Acceleration<'ctx> {
    raw: optix_sys::OptixDeviceContext,
    table: &'ctx FunctionTable,
}

impl<'ctx> Acceleration<'ctx> {
    pub(super) fn new(raw: optix_sys::OptixDeviceContext, table: &'ctx FunctionTable) -> Self {
        Self { raw, table }
    }

    pub(crate) fn compute_memory_usage(
        &self,
        options: &optix_sys::OptixAccelBuildOptions,
        inputs: &[optix_sys::OptixBuildInput],
    ) -> Result<optix_sys::OptixAccelBufferSizes> {
        let mut sizes = optix_sys::OptixAccelBufferSizes::default();
        error::check(unsafe {
            (self.table.raw.optixAccelComputeMemoryUsage.unwrap())(
                self.raw,
                options,
                inputs.as_ptr(),
                inputs.len() as u32,
                &mut sizes,
            )
        })?;
        Ok(sizes)
    }

    pub(crate) fn build(
        &self,
        stream: optix_sys::CUstream,
        options: &optix_sys::OptixAccelBuildOptions,
        inputs: &[optix_sys::OptixBuildInput],
        temp_buffer: optix_sys::CUdeviceptr,
        temp_buffer_size: usize,
        output_buffer: optix_sys::CUdeviceptr,
        output_buffer_size: usize,
    ) -> Result<optix_sys::OptixTraversableHandle> {
        let mut handle = 0;
        error::check(unsafe {
            (self.table.raw.optixAccelBuild.unwrap())(
                self.raw,
                stream,
                options,
                inputs.as_ptr(),
                inputs.len() as u32,
                temp_buffer,
                temp_buffer_size,
                output_buffer,
                output_buffer_size,
                &mut handle,
                ptr::null(),
                0,
            )
        })?;
        Ok(handle)
    }

    pub(crate) fn compact(
        &self,
        stream: optix_sys::CUstream,
        input_handle: optix_sys::OptixTraversableHandle,
        output_buffer: optix_sys::CUdeviceptr,
        output_buffer_size: usize,
    ) -> Result<optix_sys::OptixTraversableHandle> {
        let mut handle = 0;
        error::check(unsafe {
            (self.table.raw.optixAccelCompact.unwrap())(
                self.raw,
                stream,
                input_handle,
                output_buffer,
                output_buffer_size,
                &mut handle,
            )
        })?;
        Ok(handle)
    }
}
