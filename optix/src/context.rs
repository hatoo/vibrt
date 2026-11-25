use crate::error::{self, OptixError, Result};
use crate::sys::FunctionTable;
use crate::types::{DeviceProperty, ValidationMode};
use std::ffi::CString;
use std::ptr;
use std::sync::Arc;

mod acceleration;
mod resources;

struct LogCallback(Box<dyn Fn(u32, &str, &str) + Send>);

impl LogCallback {
    unsafe extern "C" fn call(
        level: std::os::raw::c_uint,
        tag: *const std::os::raw::c_char,
        message: *const std::os::raw::c_char,
        data: *mut std::ffi::c_void,
    ) {
        let tag = std::ffi::CStr::from_ptr(tag).to_string_lossy();
        let message = std::ffi::CStr::from_ptr(message).to_string_lossy();
        let callback = &*data.cast::<Self>();
        (callback.0)(level, &tag, &message);
    }
}

/// Options for creating a device context.
#[derive(Debug, Default)]
pub struct DeviceContextOptions {
    pub validation_mode: ValidationMode,
}

/// RAII wrapper around an OptiX device context.
///
/// Manages the lifetime of an `OptixDeviceContext`. When dropped,
/// the underlying context is destroyed.
pub struct DeviceContext {
    raw: optix_sys::OptixDeviceContext,
    table: Arc<FunctionTable>,
    /// OptiX retains a pointer to this allocation until replacement or destruction.
    log_callback: Option<Box<LogCallback>>,
}

unsafe impl Send for DeviceContext {}
unsafe impl Sync for DeviceContext {}

impl DeviceContext {
    /// Acceleration operations borrow the context for their complete FFI call.
    pub(crate) fn acceleration(&self) -> acceleration::Acceleration<'_> {
        acceleration::Acceleration::new(self.raw, &self.table)
    }

    /// Create a new device context from a CUDA context.
    pub fn new(
        optix: &crate::Optix,
        cuda_context: optix_sys::CUcontext,
        options: &DeviceContextOptions,
    ) -> Result<Self> {
        let raw_options = optix_sys::OptixDeviceContextOptions {
            logCallbackFunction: None,
            logCallbackData: ptr::null_mut(),
            logCallbackLevel: 0,
            validationMode: options.validation_mode.to_raw(),
        };

        let mut raw: optix_sys::OptixDeviceContext = ptr::null_mut();
        let result = unsafe {
            (optix.table.raw.optixDeviceContextCreate.unwrap())(
                cuda_context,
                &raw_options,
                &mut raw,
            )
        };
        error::check(result)?;

        Ok(Self {
            raw,
            table: optix.table.clone(),
            log_callback: None,
        })
    }

    /// Query a device property.
    pub fn get_property(&self, property: DeviceProperty) -> Result<u32> {
        let mut value: u32 = 0;
        let result = unsafe {
            (self.table.raw.optixDeviceContextGetProperty.unwrap())(
                self.raw,
                property.to_raw(),
                &mut value as *mut u32 as *mut std::ffi::c_void,
                std::mem::size_of::<u32>(),
            )
        };
        error::check(result)?;
        Ok(value)
    }

    /// Set the log callback for this context.
    pub fn set_log_callback<F>(&mut self, callback: F, level: u32) -> Result<()>
    where
        F: Fn(u32, &str, &str) + Send + 'static,
    {
        let mut callback = Box::new(LogCallback(Box::new(callback)));
        let cb_ptr = (&mut *callback as *mut LogCallback).cast();

        let result = unsafe {
            (self.table.raw.optixDeviceContextSetLogCallback.unwrap())(
                self.raw,
                Some(LogCallback::call),
                cb_ptr,
                level,
            )
        };
        error::check(result)?;
        // Keep the allocation, not just the closure inside it. On failure the
        // new callback is dropped and the previously registered one stays alive.
        self.log_callback = Some(callback);
        Ok(())
    }

    /// Enable or disable the disk cache.
    pub fn set_cache_enabled(&self, enabled: bool) -> Result<()> {
        let result = unsafe {
            (self.table.raw.optixDeviceContextSetCacheEnabled.unwrap())(self.raw, enabled as i32)
        };
        error::check(result)
    }

    /// Set the disk cache location.
    pub fn set_cache_location(&self, path: &str) -> Result<()> {
        let cpath = CString::new(path).map_err(|_| OptixError::InvalidValue)?;
        let result = unsafe {
            (self.table.raw.optixDeviceContextSetCacheLocation.unwrap())(self.raw, cpath.as_ptr())
        };
        error::check(result)
    }

    /// Set the disk cache size limits.
    pub fn set_cache_database_sizes(
        &self,
        low_water_mark: usize,
        high_water_mark: usize,
    ) -> Result<()> {
        let result = unsafe {
            (self
                .table
                .raw
                .optixDeviceContextSetCacheDatabaseSizes
                .unwrap())(self.raw, low_water_mark, high_water_mark)
        };
        error::check(result)
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            (self.table.raw.optixDeviceContextDestroy.unwrap())(self.raw);
        }
    }
}

#[cfg(test)]
mod tests;
