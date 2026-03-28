//! Minimal CUDA Driver API bindings loaded dynamically.
//!
//! Only the functions needed by the OptiX example are included here.

use std::ffi::c_void;

pub type CUdevice = i32;
pub type CUresult = u32;

pub const CUDA_SUCCESS: CUresult = 0;

pub struct CudaApi {
    _lib: libloading::Library,
    pub cuInit: unsafe extern "system" fn(flags: u32) -> CUresult,
    pub cuDeviceGet: unsafe extern "system" fn(device: *mut CUdevice, ordinal: i32) -> CUresult,
    pub cuCtxCreate_v2:
        unsafe extern "system" fn(pctx: *mut super::CUcontext, flags: u32, dev: CUdevice) -> CUresult,
    pub cuCtxDestroy_v2: unsafe extern "system" fn(ctx: super::CUcontext) -> CUresult,
    pub cuMemAlloc_v2:
        unsafe extern "system" fn(dptr: *mut super::CUdeviceptr, bytesize: usize) -> CUresult,
    pub cuMemFree_v2: unsafe extern "system" fn(dptr: super::CUdeviceptr) -> CUresult,
    pub cuMemcpyDtoH_v2: unsafe extern "system" fn(
        dstHost: *mut c_void,
        srcDevice: super::CUdeviceptr,
        byteCount: usize,
    ) -> CUresult,
    pub cuStreamCreate:
        unsafe extern "system" fn(phStream: *mut super::CUstream, flags: u32) -> CUresult,
    pub cuStreamDestroy_v2: unsafe extern "system" fn(hStream: super::CUstream) -> CUresult,
    pub cuMemcpyHtoD_v2: unsafe extern "system" fn(
        dstDevice: super::CUdeviceptr,
        srcHost: *const c_void,
        byteCount: usize,
    ) -> CUresult,
    pub cuStreamSynchronize: unsafe extern "system" fn(hStream: super::CUstream) -> CUresult,
}

macro_rules! load_sym {
    ($lib:expr, $name:expr) => {
        *$lib
            .get::<_>($name)
            .map_err(|e| format!("Failed to load {}: {}", std::str::from_utf8($name).unwrap_or("?"), e))?
    };
}

impl CudaApi {
    pub fn load() -> Result<Self, String> {
        #[cfg(target_os = "windows")]
        let lib = unsafe { libloading::Library::new("nvcuda.dll") }
            .map_err(|e| format!("Failed to load nvcuda.dll: {}", e))?;

        #[cfg(target_os = "linux")]
        let lib = unsafe { libloading::Library::new("libcuda.so.1") }
            .map_err(|e| format!("Failed to load libcuda.so.1: {}", e))?;

        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        return Err("Unsupported platform".to_string());

        unsafe {
            Ok(CudaApi {
                cuInit: load_sym!(lib, b"cuInit\0"),
                cuDeviceGet: load_sym!(lib, b"cuDeviceGet\0"),
                cuCtxCreate_v2: load_sym!(lib, b"cuCtxCreate_v2\0"),
                cuCtxDestroy_v2: load_sym!(lib, b"cuCtxDestroy_v2\0"),
                cuMemAlloc_v2: load_sym!(lib, b"cuMemAlloc_v2\0"),
                cuMemFree_v2: load_sym!(lib, b"cuMemFree_v2\0"),
                cuMemcpyDtoH_v2: load_sym!(lib, b"cuMemcpyDtoH_v2\0"),
                cuMemcpyHtoD_v2: load_sym!(lib, b"cuMemcpyHtoD_v2\0"),
                cuStreamCreate: load_sym!(lib, b"cuStreamCreate\0"),
                cuStreamDestroy_v2: load_sym!(lib, b"cuStreamDestroy_v2\0"),
                cuStreamSynchronize: load_sym!(lib, b"cuStreamSynchronize\0"),
                _lib: lib,
            })
        }
    }
}
