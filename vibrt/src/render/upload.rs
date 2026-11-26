//! Owned raw GPU uploads shared by scene preparation and image jobs.
use crate::CudaResultExt;
use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::{mem, sync::Arc};
pub(super) fn dptr_u8(s: &CudaSlice<u8>, stream: &CudaStream) -> optix_sys::CUdeviceptr {
    let (p, _) = s.device_ptr(stream);
    p as optix_sys::CUdeviceptr
}
pub(super) fn dptr_u32(s: &CudaSlice<u32>, stream: &CudaStream) -> optix_sys::CUdeviceptr {
    let (p, _) = s.device_ptr(stream);
    p as optix_sys::CUdeviceptr
}
pub(super) fn dptr_f32(s: &CudaSlice<f32>, stream: &CudaStream) -> optix_sys::CUdeviceptr {
    let (p, _) = s.device_ptr(stream);
    p as optix_sys::CUdeviceptr
}

/// Own every raw ABI upload until the render ends, including error/cancel paths.
/// Keeping just its integer device address leaks memory across Blender renders.
pub(super) struct UploadArena {
    pub(super) stream: Arc<CudaStream>,
    pub(super) buffers: std::cell::RefCell<Vec<CudaSlice<u8>>>,
}
impl UploadArena {
    pub(super) fn copy<T>(&self, value: &T) -> Result<optix_sys::CUdeviceptr> {
        self.copy_slice(std::slice::from_ref(value))
    }
    pub(super) fn copy_slice<T>(&self, data: &[T]) -> Result<optix_sys::CUdeviceptr> {
        let size = mem::size_of_val(data);
        if size == 0 {
            return Ok(0);
        }
        // Callers supply repr(C) GPU structs or primitive arrays.
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size) };
        let buffer = self.stream.clone_htod(bytes).cuda()?;
        let pointer = dptr_u8(&buffer, &self.stream);
        self.buffers.borrow_mut().push(buffer);
        Ok(pointer)
    }
}
