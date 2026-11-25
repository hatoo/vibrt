use super::*;
use crate::{Module, ModuleCompileOptions, PipelineCompileOptions};
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Default)]
struct Backend {
    callback: optix_sys::OptixLogCallback,
    callback_data: *mut c_void,
    reject_callback: bool,
    reject_module: bool,
    destroyed: bool,
    modules_destroyed: usize,
    source: Vec<u8>,
    params_name: String,
    payload_count: i32,
}

impl Backend {
    fn emit(&self) {
        if let Some(callback) = self.callback {
            unsafe { callback(3, c"test".as_ptr(), c"message".as_ptr(), self.callback_data) };
        }
    }
}

unsafe extern "C" fn set_callback(
    raw: optix_sys::OptixDeviceContext,
    callback: optix_sys::OptixLogCallback,
    data: *mut c_void,
    _level: u32,
) -> optix_sys::OptixResult {
    let state = &mut *raw.cast::<Backend>();
    if state.reject_callback {
        return optix_sys::OptixResult::OPTIX_ERROR_INVALID_VALUE;
    }
    state.callback = callback;
    state.callback_data = data;
    optix_sys::OptixResult::OPTIX_SUCCESS
}

unsafe extern "C" fn destroy_context(raw: optix_sys::OptixDeviceContext) -> optix_sys::OptixResult {
    let state = &mut *raw.cast::<Backend>();
    state.emit();
    state.destroyed = true;
    optix_sys::OptixResult::OPTIX_SUCCESS
}

unsafe extern "C" fn create_module(
    raw: optix_sys::OptixDeviceContext,
    _options: *const optix_sys::OptixModuleCompileOptions,
    pipeline: *const optix_sys::OptixPipelineCompileOptions,
    source: *const i8,
    source_size: usize,
    log: *mut i8,
    log_size: *mut usize,
    module: *mut optix_sys::OptixModule,
) -> optix_sys::OptixResult {
    let state = &mut *raw.cast::<Backend>();
    if state.reject_module {
        return optix_sys::OptixResult::OPTIX_ERROR_INVALID_VALUE;
    }
    state.source = std::slice::from_raw_parts(source.cast(), source_size).to_vec();
    state.params_name = CStr::from_ptr((*pipeline).pipelineLaunchParamsVariableName)
        .to_string_lossy()
        .into_owned();
    state.payload_count = (*pipeline).numPayloadValues;
    let message = c"compiled".to_bytes_with_nul();
    ptr::copy_nonoverlapping(message.as_ptr(), log.cast(), message.len());
    *log_size = message.len();
    *module = raw.cast();
    optix_sys::OptixResult::OPTIX_SUCCESS
}

unsafe extern "C" fn destroy_module(raw: optix_sys::OptixModule) -> optix_sys::OptixResult {
    (*raw.cast::<Backend>()).modules_destroyed += 1;
    optix_sys::OptixResult::OPTIX_SUCCESS
}

fn context(state: &mut Backend) -> DeviceContext {
    let table = optix_sys::OptixFunctionTable {
        optixDeviceContextSetLogCallback: Some(set_callback),
        optixDeviceContextDestroy: Some(destroy_context),
        optixModuleCreate: Some(create_module),
        optixModuleDestroy: Some(destroy_module),
        ..Default::default()
    };
    DeviceContext {
        raw: (state as *mut Backend).cast(),
        table: FunctionTable::new(table),
        log_callback: None,
    }
}

struct DropCounter(Arc<AtomicUsize>);

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn callback_survives_registration_context_moves_and_destruction() {
    let mut backend = Backend::default();
    let mut ctx = context(&mut backend);
    let messages = Arc::new(Mutex::new(Vec::new()));
    let captured = messages.clone();
    let drops = Arc::new(AtomicUsize::new(0));
    let lifetime = DropCounter(drops.clone());
    ctx.set_log_callback(
        move |level, tag, message| {
            let _ = &lifetime;
            captured
                .lock()
                .unwrap()
                .push((level, tag.to_owned(), message.to_owned()));
        },
        3,
    )
    .unwrap();
    let ctx = Box::new(ctx);
    backend.emit();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(ctx);
    assert!(backend.destroyed);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(
        *messages.lock().unwrap(),
        vec![(3, "test".into(), "message".into()); 2]
    );
}

#[test]
fn callback_replacement_keeps_old_registration_on_failure_and_releases_both() {
    let mut backend = Backend::default();
    let mut ctx = context(&mut backend);
    let calls = Arc::new(AtomicUsize::new(0));
    let captured = calls.clone();
    let old_drops = Arc::new(AtomicUsize::new(0));
    let old_lifetime = DropCounter(old_drops.clone());
    ctx.set_log_callback(
        move |_, _, _| {
            let _ = &old_lifetime;
            captured.fetch_add(1, Ordering::SeqCst);
        },
        3,
    )
    .unwrap();
    let failed_drops = Arc::new(AtomicUsize::new(0));
    let failed_lifetime = DropCounter(failed_drops.clone());
    backend.reject_callback = true;
    assert!(matches!(
        ctx.set_log_callback(
            move |_, _, _| {
                let _ = &failed_lifetime;
            },
            3
        ),
        Err(OptixError::InvalidValue)
    ));
    assert_eq!(failed_drops.load(Ordering::SeqCst), 1);
    assert_eq!(old_drops.load(Ordering::SeqCst), 0);
    backend.emit();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    backend.reject_callback = false;
    ctx.set_log_callback(|_, _, _| {}, 3).unwrap();
    assert_eq!(old_drops.load(Ordering::SeqCst), 1);
    backend.emit();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn module_factory_preserves_options_source_logs_and_destruction() {
    let mut backend = Backend::default();
    let ctx = context(&mut backend);
    let options = PipelineCompileOptions::new("launch_params").num_payload_values(7);
    let compiled = Module::new(
        &ctx,
        &ModuleCompileOptions::default(),
        &options,
        b"ptx\0bytes",
    )
    .unwrap();
    assert_eq!(compiled.log, "compiled");
    assert_eq!(backend.source, b"ptx\0bytes");
    assert_eq!(backend.params_name, "launch_params");
    assert_eq!(backend.payload_count, 7);
    assert_eq!(backend.modules_destroyed, 0);
    drop(compiled.value);
    assert_eq!(backend.modules_destroyed, 1);
}

#[test]
fn module_factory_propagates_failure_without_creating_a_resource() {
    let mut backend = Backend {
        reject_module: true,
        ..Default::default()
    };
    let ctx = context(&mut backend);
    let result = Module::new(
        &ctx,
        &ModuleCompileOptions::default(),
        &PipelineCompileOptions::new("params"),
        b"invalid",
    );
    assert!(matches!(result, Err(OptixError::InvalidValue)));
    assert_eq!(backend.modules_destroyed, 0);
}
