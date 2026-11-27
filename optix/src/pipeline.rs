pub use crate::compile_options::PipelineCompileOptions;
use crate::context::DeviceContext;
use crate::error::{self, Result, WithLog};
use crate::program_group::ProgramGroup;
use crate::sys::FunctionTable;
use std::sync::Arc;

/// Options for pipeline linking.
#[derive(Debug, Clone)]
pub struct PipelineLinkOptions {
    pub max_trace_depth: u32,
}

impl Default for PipelineLinkOptions {
    fn default() -> Self {
        Self { max_trace_depth: 1 }
    }
}

/// RAII wrapper around an OptiX pipeline.
pub struct Pipeline {
    pub(crate) raw: optix_sys::OptixPipeline,
    pub(crate) table: Arc<FunctionTable>,
}

unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

impl Pipeline {
    /// Create a new pipeline from program groups.
    pub fn new(
        ctx: &DeviceContext,
        compile_options: &PipelineCompileOptions,
        link_options: &PipelineLinkOptions,
        program_groups: &[&ProgramGroup],
    ) -> Result<WithLog<Self>> {
        let raw_compile = compile_options.to_raw();
        let raw_link = optix_sys::OptixPipelineLinkOptions {
            maxTraceDepth: link_options.max_trace_depth,
        };

        let raw_pgs: Vec<optix_sys::OptixProgramGroup> =
            program_groups.iter().map(|pg| pg.raw).collect();

        let WithLog {
            value: (raw, table),
            log,
        } = ctx.create_pipeline(&raw_compile, &raw_link, &raw_pgs)?;
        Ok(WithLog {
            value: Self { raw, table },
            log,
        })
    }

    /// Set the stack sizes for the pipeline.
    /// Compute actual program requirements instead of guessing a fixed stack.
    /// This is optixUtilComputeStackSizes with continuation callable program groups (no direct callables).
    pub fn set_stack_size_for_programs(
        &self,
        groups: &[&ProgramGroup],
        max_trace_depth: u32,
        max_continuation_callable_depth: u32,
        max_traversable_graph_depth: u32,
    ) -> Result<u32> {
        let mut sizes: optix_sys::OptixStackSizes = unsafe { std::mem::zeroed() };
        for group in groups {
            let mut current: optix_sys::OptixStackSizes = unsafe { std::mem::zeroed() };
            error::check(unsafe {
                (self.table.raw.optixProgramGroupGetStackSize.unwrap())(
                    group.raw,
                    &mut current,
                    self.raw,
                )
            })?;
            sizes.cssRG = sizes.cssRG.max(current.cssRG);
            sizes.cssMS = sizes.cssMS.max(current.cssMS);
            sizes.cssCH = sizes.cssCH.max(current.cssCH);
            sizes.cssAH = sizes.cssAH.max(current.cssAH);
            sizes.cssIS = sizes.cssIS.max(current.cssIS);
            sizes.cssCC = sizes.cssCC.max(current.cssCC);
        }
        let callable_tree = max_continuation_callable_depth * sizes.cssCC;
        let hit_or_miss = sizes.cssCH.max(sizes.cssMS) + callable_tree;
        let continuation = sizes.cssRG
            + callable_tree
            + max_trace_depth.saturating_sub(1) * hit_or_miss
            + max_trace_depth.min(1) * hit_or_miss.max(sizes.cssIS + sizes.cssAH);
        self.set_stack_size(0, 0, continuation, max_traversable_graph_depth)?;
        Ok(continuation)
    }

    pub fn set_stack_size(
        &self,
        direct_callable_from_traversal: u32,
        direct_callable_from_state: u32,
        continuation: u32,
        max_traversable_graph_depth: u32,
    ) -> Result<()> {
        let result = unsafe {
            (self.table.raw.optixPipelineSetStackSize.unwrap())(
                self.raw,
                direct_callable_from_traversal,
                direct_callable_from_state,
                continuation,
                max_traversable_graph_depth,
            )
        };
        error::check(result)
    }

    /// Launch the pipeline.
    pub fn launch(
        &self,
        stream: optix_sys::CUstream,
        params_device_ptr: optix_sys::CUdeviceptr,
        params_size: usize,
        sbt: &optix_sys::OptixShaderBindingTable,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Result<()> {
        let result = unsafe {
            (self.table.raw.optixLaunch.unwrap())(
                self.raw,
                stream,
                params_device_ptr,
                params_size,
                sbt,
                width,
                height,
                depth,
            )
        };
        error::check(result)
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            (self.table.raw.optixPipelineDestroy.unwrap())(self.raw);
        }
    }
}
