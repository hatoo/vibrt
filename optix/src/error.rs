use std::fmt;

/// All OptiX error codes as a proper Rust enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OptixError {
    InvalidValue,
    HostOutOfMemory,
    InvalidOperation,
    FileIoError,
    InvalidFileFormat,
    DiskCacheInvalidPath,
    DiskCachePermissionError,
    DiskCacheDatabaseError,
    DiskCacheInvalidData,
    LaunchFailure,
    InvalidDeviceContext,
    CudaNotInitialized,
    ValidationFailure,
    InvalidInput,
    InvalidLaunchParameter,
    InvalidPayloadAccess,
    InvalidAttributeAccess,
    InvalidFunctionUse,
    InvalidFunctionArguments,
    PipelineOutOfConstantMemory,
    PipelineLinkError,
    IllegalDuringTaskExecute,
    InternalCompilerError,
    DenoiserModelNotSet,
    DenoiserNotInitialized,
    NotCompatible,
    PayloadTypeMismatch,
    PayloadTypeResolutionFailed,
    PayloadTypeIdInvalid,
    NotSupported,
    UnsupportedAbiVersion,
    FunctionTableSizeMismatch,
    InvalidEntryFunctionOptions,
    LibraryNotFound,
    EntrySymbolNotFound,
    LibraryUnloadFailure,
    DeviceOutOfMemory,
    InvalidPointer,
    CudaError,
    InternalError,
    Unknown(i32),
}

impl OptixError {
    pub(crate) fn from_raw(r: optix_sys::OptixResult) -> Self {
        use optix_sys::OptixResult as R;
        match r {
            R::OPTIX_ERROR_INVALID_VALUE => Self::InvalidValue,
            R::OPTIX_ERROR_HOST_OUT_OF_MEMORY => Self::HostOutOfMemory,
            R::OPTIX_ERROR_INVALID_OPERATION => Self::InvalidOperation,
            R::OPTIX_ERROR_FILE_IO_ERROR => Self::FileIoError,
            R::OPTIX_ERROR_INVALID_FILE_FORMAT => Self::InvalidFileFormat,
            R::OPTIX_ERROR_DISK_CACHE_INVALID_PATH => Self::DiskCacheInvalidPath,
            R::OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR => Self::DiskCachePermissionError,
            R::OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR => Self::DiskCacheDatabaseError,
            R::OPTIX_ERROR_DISK_CACHE_INVALID_DATA => Self::DiskCacheInvalidData,
            R::OPTIX_ERROR_LAUNCH_FAILURE => Self::LaunchFailure,
            R::OPTIX_ERROR_INVALID_DEVICE_CONTEXT => Self::InvalidDeviceContext,
            R::OPTIX_ERROR_CUDA_NOT_INITIALIZED => Self::CudaNotInitialized,
            R::OPTIX_ERROR_VALIDATION_FAILURE => Self::ValidationFailure,
            R::OPTIX_ERROR_INVALID_INPUT => Self::InvalidInput,
            R::OPTIX_ERROR_INVALID_LAUNCH_PARAMETER => Self::InvalidLaunchParameter,
            R::OPTIX_ERROR_INVALID_PAYLOAD_ACCESS => Self::InvalidPayloadAccess,
            R::OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS => Self::InvalidAttributeAccess,
            R::OPTIX_ERROR_INVALID_FUNCTION_USE => Self::InvalidFunctionUse,
            R::OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS => Self::InvalidFunctionArguments,
            R::OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY => Self::PipelineOutOfConstantMemory,
            R::OPTIX_ERROR_PIPELINE_LINK_ERROR => Self::PipelineLinkError,
            R::OPTIX_ERROR_ILLEGAL_DURING_TASK_EXECUTE => Self::IllegalDuringTaskExecute,
            R::OPTIX_ERROR_INTERNAL_COMPILER_ERROR => Self::InternalCompilerError,
            R::OPTIX_ERROR_DENOISER_MODEL_NOT_SET => Self::DenoiserModelNotSet,
            R::OPTIX_ERROR_DENOISER_NOT_INITIALIZED => Self::DenoiserNotInitialized,
            R::OPTIX_ERROR_NOT_COMPATIBLE => Self::NotCompatible,
            R::OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH => Self::PayloadTypeMismatch,
            R::OPTIX_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED => Self::PayloadTypeResolutionFailed,
            R::OPTIX_ERROR_PAYLOAD_TYPE_ID_INVALID => Self::PayloadTypeIdInvalid,
            R::OPTIX_ERROR_NOT_SUPPORTED => Self::NotSupported,
            R::OPTIX_ERROR_UNSUPPORTED_ABI_VERSION => Self::UnsupportedAbiVersion,
            R::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH => Self::FunctionTableSizeMismatch,
            R::OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS => Self::InvalidEntryFunctionOptions,
            R::OPTIX_ERROR_LIBRARY_NOT_FOUND => Self::LibraryNotFound,
            R::OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND => Self::EntrySymbolNotFound,
            R::OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE => Self::LibraryUnloadFailure,
            R::OPTIX_ERROR_DEVICE_OUT_OF_MEMORY => Self::DeviceOutOfMemory,
            R::OPTIX_ERROR_INVALID_POINTER => Self::InvalidPointer,
            R::OPTIX_ERROR_CUDA_ERROR => Self::CudaError,
            R::OPTIX_ERROR_INTERNAL_ERROR => Self::InternalError,
            other => Self::Unknown(other.0),
        }
    }
}

impl fmt::Display for OptixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidValue => write!(f, "invalid value"),
            Self::HostOutOfMemory => write!(f, "host out of memory"),
            Self::InvalidOperation => write!(f, "invalid operation"),
            Self::FileIoError => write!(f, "file I/O error"),
            Self::InvalidFileFormat => write!(f, "invalid file format"),
            Self::DiskCacheInvalidPath => write!(f, "disk cache invalid path"),
            Self::DiskCachePermissionError => write!(f, "disk cache permission error"),
            Self::DiskCacheDatabaseError => write!(f, "disk cache database error"),
            Self::DiskCacheInvalidData => write!(f, "disk cache invalid data"),
            Self::LaunchFailure => write!(f, "launch failure"),
            Self::InvalidDeviceContext => write!(f, "invalid device context"),
            Self::CudaNotInitialized => write!(f, "CUDA not initialized"),
            Self::ValidationFailure => write!(f, "validation failure"),
            Self::InvalidInput => write!(f, "invalid input"),
            Self::InvalidLaunchParameter => write!(f, "invalid launch parameter"),
            Self::InvalidPayloadAccess => write!(f, "invalid payload access"),
            Self::InvalidAttributeAccess => write!(f, "invalid attribute access"),
            Self::InvalidFunctionUse => write!(f, "invalid function use"),
            Self::InvalidFunctionArguments => write!(f, "invalid function arguments"),
            Self::PipelineOutOfConstantMemory => write!(f, "pipeline out of constant memory"),
            Self::PipelineLinkError => write!(f, "pipeline link error"),
            Self::IllegalDuringTaskExecute => write!(f, "illegal during task execute"),
            Self::InternalCompilerError => write!(f, "internal compiler error"),
            Self::DenoiserModelNotSet => write!(f, "denoiser model not set"),
            Self::DenoiserNotInitialized => write!(f, "denoiser not initialized"),
            Self::NotCompatible => write!(f, "not compatible"),
            Self::PayloadTypeMismatch => write!(f, "payload type mismatch"),
            Self::PayloadTypeResolutionFailed => write!(f, "payload type resolution failed"),
            Self::PayloadTypeIdInvalid => write!(f, "payload type ID invalid"),
            Self::NotSupported => write!(f, "not supported"),
            Self::UnsupportedAbiVersion => write!(f, "unsupported ABI version"),
            Self::FunctionTableSizeMismatch => write!(f, "function table size mismatch"),
            Self::InvalidEntryFunctionOptions => write!(f, "invalid entry function options"),
            Self::LibraryNotFound => write!(f, "library not found"),
            Self::EntrySymbolNotFound => write!(f, "entry symbol not found"),
            Self::LibraryUnloadFailure => write!(f, "library unload failure"),
            Self::DeviceOutOfMemory => write!(f, "device out of memory"),
            Self::InvalidPointer => write!(f, "invalid pointer"),
            Self::CudaError => write!(f, "CUDA error"),
            Self::InternalError => write!(f, "internal error"),
            Self::Unknown(code) => write!(f, "unknown OptiX error ({})", code),
        }
    }
}

impl std::error::Error for OptixError {}

/// Alias for `std::result::Result<T, OptixError>`.
pub type Result<T> = std::result::Result<T, OptixError>;

/// Wraps a value alongside its OptiX compilation/link log.
#[derive(Debug)]
pub struct WithLog<T> {
    pub value: T,
    pub log: String,
}

/// Check an OptixResult and convert to Result<()>.
pub(crate) fn check(result: optix_sys::OptixResult) -> Result<()> {
    if result == optix_sys::OptixResult::OPTIX_SUCCESS {
        Ok(())
    } else {
        Err(OptixError::from_raw(result))
    }
}

/// Extract the log string from a raw log buffer.
pub(crate) fn extract_log(buf: &[u8], size: usize) -> String {
    let len = size.min(buf.len()).saturating_sub(1); // exclude null terminator
    String::from_utf8_lossy(&buf[..len]).to_string()
}
