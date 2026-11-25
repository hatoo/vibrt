//! Context-bound OptiX calls. Callers provide descriptors; the context owns
//! dispatch, error checking, logs, and sharing the table with new resources.

use super::DeviceContext;
use crate::error::{self, Result, WithLog};
use crate::sys::FunctionTable;
use std::ptr;
use std::sync::Arc;

type Resource<T> = (T, Arc<FunctionTable>);

impl DeviceContext {
    fn create_resource<T: Default>(
        &self,
        create: impl FnOnce(&mut T) -> optix_sys::OptixResult,
    ) -> Result<Resource<T>> {
        let mut raw = T::default();
        error::check(create(&mut raw))?;
        Ok((raw, self.table.clone()))
    }

    fn create_logged_resource<T: Default>(
        &self,
        create: impl FnOnce(&mut T, *mut i8, &mut usize) -> optix_sys::OptixResult,
    ) -> Result<WithLog<Resource<T>>> {
        let mut log = [0u8; 2048];
        let mut log_size = log.len();
        let value =
            self.create_resource(|raw| create(raw, log.as_mut_ptr().cast(), &mut log_size))?;
        Ok(WithLog {
            value,
            log: error::extract_log(&log, log_size),
        })
    }

    pub(crate) fn create_module(
        &self,
        options: &optix_sys::OptixModuleCompileOptions,
        pipeline_options: &optix_sys::OptixPipelineCompileOptions,
        source: &[u8],
    ) -> Result<WithLog<Resource<optix_sys::OptixModule>>> {
        self.create_logged_resource(|raw, log, log_size| unsafe {
            (self.table.raw.optixModuleCreate.unwrap())(
                self.raw,
                options,
                pipeline_options,
                source.as_ptr().cast(),
                source.len(),
                log,
                log_size,
                raw,
            )
        })
    }

    pub(crate) fn builtin_is_module(
        &self,
        options: &optix_sys::OptixModuleCompileOptions,
        pipeline_options: &optix_sys::OptixPipelineCompileOptions,
        builtin_options: &optix_sys::OptixBuiltinISOptions,
    ) -> Result<Resource<optix_sys::OptixModule>> {
        self.create_resource(|raw| unsafe {
            (self.table.raw.optixBuiltinISModuleGet.unwrap())(
                self.raw,
                options,
                pipeline_options,
                builtin_options,
                raw,
            )
        })
    }

    pub(crate) fn create_program_group(
        &self,
        desc: &optix_sys::OptixProgramGroupDesc,
    ) -> Result<WithLog<Resource<optix_sys::OptixProgramGroup>>> {
        let options = optix_sys::OptixProgramGroupOptions {
            payloadType: ptr::null(),
        };
        self.create_logged_resource(|raw, log, log_size| unsafe {
            (self.table.raw.optixProgramGroupCreate.unwrap())(
                self.raw, desc, 1, &options, log, log_size, raw,
            )
        })
    }

    pub(crate) fn create_pipeline(
        &self,
        compile_options: &optix_sys::OptixPipelineCompileOptions,
        link_options: &optix_sys::OptixPipelineLinkOptions,
        groups: &[optix_sys::OptixProgramGroup],
    ) -> Result<WithLog<Resource<optix_sys::OptixPipeline>>> {
        self.create_logged_resource(|raw, log, log_size| unsafe {
            (self.table.raw.optixPipelineCreate.unwrap())(
                self.raw,
                compile_options,
                link_options,
                groups.as_ptr(),
                groups.len() as u32,
                log,
                log_size,
                raw,
            )
        })
    }

    pub(crate) fn create_denoiser(
        &self,
        model_kind: optix_sys::OptixDenoiserModelKind,
        options: &optix_sys::OptixDenoiserOptions,
    ) -> Result<Resource<optix_sys::OptixDenoiser>> {
        self.create_resource(|raw| unsafe {
            (self.table.raw.optixDenoiserCreate.unwrap())(self.raw, model_kind, options, raw)
        })
    }
}
