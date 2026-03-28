use std::sync::Arc;

/// Internal wrapper around the raw OptiX function table.
pub(crate) struct FunctionTable {
    pub(crate) raw: optix_sys::OptixFunctionTable,
}

impl FunctionTable {
    pub(crate) fn new(raw: optix_sys::OptixFunctionTable) -> Arc<Self> {
        Arc::new(Self { raw })
    }
}
