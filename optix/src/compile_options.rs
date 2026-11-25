//! Compilation settings shared by modules and pipelines.

use crate::types::{ExceptionFlags, PrimitiveTypeFlags, TraversableGraphFlags};
use std::ffi::CString;

/// Options for pipeline compilation.
///
/// This struct owns the launch params variable name string.
pub struct PipelineCompileOptions {
    pub uses_motion_blur: bool,
    pub traversable_graph_flags: TraversableGraphFlags,
    pub num_payload_values: i32,
    pub num_attribute_values: i32,
    pub exception_flags: ExceptionFlags,
    pub uses_primitive_type_flags: PrimitiveTypeFlags,
    pub allow_opacity_micromaps: bool,
    launch_params_name: CString,
}

impl PipelineCompileOptions {
    /// Create pipeline compile options with the given launch params variable name.
    ///
    /// The name must match the `extern "C" { __constant__ Params params; }` declaration
    /// in your device code.
    pub fn new(launch_params_name: &str) -> Self {
        Self {
            uses_motion_blur: false,
            traversable_graph_flags: TraversableGraphFlags::ALLOW_SINGLE_GAS,
            num_payload_values: 3,
            num_attribute_values: 3,
            exception_flags: ExceptionFlags::NONE,
            uses_primitive_type_flags: PrimitiveTypeFlags::TRIANGLE,
            allow_opacity_micromaps: false,
            launch_params_name: CString::new(launch_params_name)
                .expect("invalid launch params name"),
        }
    }

    pub fn uses_motion_blur(mut self, v: bool) -> Self {
        self.uses_motion_blur = v;
        self
    }
    pub fn traversable_graph_flags(mut self, f: TraversableGraphFlags) -> Self {
        self.traversable_graph_flags = f;
        self
    }
    pub fn num_payload_values(mut self, n: i32) -> Self {
        self.num_payload_values = n;
        self
    }
    pub fn num_attribute_values(mut self, n: i32) -> Self {
        self.num_attribute_values = n;
        self
    }
    pub fn exception_flags(mut self, f: ExceptionFlags) -> Self {
        self.exception_flags = f;
        self
    }
    pub fn uses_primitive_type_flags(mut self, f: PrimitiveTypeFlags) -> Self {
        self.uses_primitive_type_flags = f;
        self
    }

    pub(crate) fn to_raw(&self) -> optix_sys::OptixPipelineCompileOptions {
        optix_sys::OptixPipelineCompileOptions {
            usesMotionBlur: self.uses_motion_blur as i32,
            traversableGraphFlags: self.traversable_graph_flags.bits(),
            numPayloadValues: self.num_payload_values,
            numAttributeValues: self.num_attribute_values,
            exceptionFlags: self.exception_flags.bits(),
            pipelineLaunchParamsVariableName: self.launch_params_name.as_ptr(),
            usesPrimitiveTypeFlags: self.uses_primitive_type_flags.bits(),
            allowOpacityMicromaps: self.allow_opacity_micromaps as i32,
            ..Default::default()
        }
    }
}
