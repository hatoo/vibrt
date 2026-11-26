//! Thread-confined GPU contexts, compiled shader variants and CPU query cache.

use super::{
    query_cache::{QueryCache, QueryTree},
    upload::dptr_u8,
};
use crate::scene_loader::LoadedMesh;
use crate::CudaResultExt;
use crate::{
    material::ShaderRequirements,
    shader_compile::{self, CompiledShaders},
    Progress,
};
use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use optix::{
    accel::{self, AccelBuildOptions, BuildInput},
    denoiser, DenoiserModelKind,
};
use optix::{DeviceContext, DeviceContextOptions};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

// Fields drop in declaration order: OptiX must go before its CUDA context.
pub struct RenderSession {
    shaders: RefCell<Option<(ShaderRequirements, Rc<CompiledShaders>)>>,
    optix: DeviceContext,
    cuda: Arc<CudaContext>,
    queries: RefCell<QueryCache>,
}

impl RenderSession {
    pub fn new() -> Result<Self> {
        let cuda = CudaContext::new(0).cuda().context("CUDA context")?;
        let handle = optix::init().context("OptiX init")?;
        let optix = DeviceContext::new(
            &handle,
            cuda.cu_ctx() as optix_sys::CUcontext,
            &DeviceContextOptions::default(),
        )
        .context("OptiX context")?;
        Ok(Self {
            shaders: RefCell::new(None),
            optix,
            cuda,
            queries: RefCell::new(QueryCache::default()),
        })
    }

    pub(super) fn stream(&self) -> Result<Arc<CudaStream>> {
        self.cuda.bind_to_thread().cuda()?;
        Ok(self.cuda.default_stream())
    }

    pub(super) fn drain(&self) -> Result<()> {
        self.cuda.bind_to_thread().cuda()?;
        self.cuda.synchronize().cuda()
    }

    pub(super) fn cleanup(&self) {
        self.cuda.record_err(self.cuda.bind_to_thread());
        self.cuda.record_err(self.cuda.synchronize());
    }

    pub(super) fn begin_queries(&self, meshes: &[LoadedMesh]) {
        self.queries.borrow_mut().begin(meshes);
    }

    pub(super) fn query_batch(&self, first: usize, meshes: &[LoadedMesh]) -> Vec<Arc<QueryTree>> {
        self.queries.borrow_mut().batch(first, meshes)
    }

    pub(super) fn build_accel(
        &self,
        options: &AccelBuildOptions,
        inputs: &[BuildInput],
    ) -> Result<BuiltAccel> {
        let stream = self.stream()?;
        let sizes = accel::accel_compute_memory_usage(&self.optix, options, inputs)?;
        let scratch = unsafe { stream.alloc(sizes.temp_size) }.cuda()?;
        let output = unsafe { stream.alloc(sizes.output_size) }.cuda()?;
        let handle = accel::accel_build(
            &self.optix,
            stream.cu_stream() as optix_sys::CUstream,
            options,
            inputs,
            dptr_u8(&scratch, &stream),
            sizes.temp_size,
            dptr_u8(&output, &stream),
            sizes.output_size,
        )?;
        Ok(BuiltAccel {
            handle,
            scratch,
            output,
        })
    }

    pub(super) fn denoiser(&self) -> Result<denoiser::Denoiser> {
        self.cuda.bind_to_thread().cuda()?;
        denoiser::Denoiser::new(
            &self.optix,
            DenoiserModelKind::Aov,
            &denoiser::DenoiserOptions {
                guide_albedo: true,
                guide_normal: true,
                ..Default::default()
            },
        )
        .context("denoiser create")
    }

    pub(super) fn shaders(
        &self,
        requirements: ShaderRequirements,
        progress: &mut dyn Progress,
    ) -> Result<(Rc<CompiledShaders>, Duration, Duration)> {
        let cached = self
            .shaders
            .borrow()
            .as_ref()
            .filter(|(key, _)| *key == requirements)
            .map(|(_, value)| value.clone());
        if let Some(shaders) = cached {
            progress.log("Reusing OptiX pipeline...");
            return Ok((shaders, Duration::ZERO, Duration::ZERO));
        }
        let shaders = Rc::new(shader_compile::compile(
            &self.optix,
            requirements,
            progress,
        )?);
        let times = (shaders.ptx_time, shaders.pipeline_time);
        *self.shaders.borrow_mut() = Some((requirements, shaders.clone()));
        Ok((shaders, times.0, times.1))
    }
}

// Scratch is retained until the build stream is drained; only output persists.
pub(super) struct BuiltAccel {
    pub handle: optix_sys::OptixTraversableHandle,
    pub scratch: CudaSlice<u8>,
    pub output: CudaSlice<u8>,
}

impl Drop for RenderSession {
    fn drop(&mut self) {
        self.cleanup();
    }
}

thread_local! {
    // Only owned query data and one shader variant accompany the contexts.
    // No scene buffers, borrowed Python arrays or callbacks are cached.
    // Each thread owns at most one idle pair, released when the thread exits.
    static IDLE: RefCell<Option<RenderSession>> = const { RefCell::new(None) };
}

pub(super) fn with_session<T>(render: impl FnOnce(&RenderSession) -> Result<T>) -> Result<T> {
    // Take the entry out while rendering. Reentrant callbacks can acquire their
    // own pair without borrowing the cache or sharing an active OptiX context.
    let contexts = match IDLE.with(|idle| idle.borrow_mut().take()) {
        Some(contexts) => contexts,
        None => RenderSession::new()?,
    };
    contexts.stream()?;
    let result = render(&contexts);
    // The render's local resources have dropped, including queued async frees.
    // Observe cleanup errors and drain work before caching or destroying OptiX.
    let drained = contexts.drain();
    if result.is_ok() && drained.is_ok() {
        let previous = IDLE.with(|idle| idle.replace(Some(contexts)));
        drop(previous);
    }
    // Errors and unwinding discard this pair rather than caching failed state.
    let output = result?;
    drained?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clear() {
        drop(IDLE.with(|idle| idle.borrow_mut().take()));
    }

    #[test]
    #[ignore = "requires CUDA and OptiX; run serially outside rendering benchmarks"]
    fn reuse_rebind_and_error_recovery() -> Result<()> {
        clear();
        let first = with_session(|contexts| Ok(contexts.cuda.clone()))?;
        // Simulate another CUDA user switching away between render callbacks.
        let other = CudaContext::new_non_primary(0, 0).cuda()?;
        with_session(|contexts| {
            assert!(Arc::ptr_eq(&first, &contexts.cuda));
            assert_eq!(
                cudarc::driver::result::ctx::get_current().cuda()?,
                Some(contexts.cuda.cu_ctx())
            );
            let stream = contexts.cuda.default_stream();
            let values = [1u32, 9, 27, 81];
            let device = stream.clone_htod(&values).cuda()?;
            assert_eq!(stream.clone_dtoh(&device).cuda()?, values);
            Ok(())
        })?;
        drop(other);
        let failed: Result<()> = with_session(|_| anyhow::bail!("cancelled callback"));
        assert!(failed.is_err());
        let recovered = with_session(|contexts| Ok(contexts.cuda.clone()))?;
        assert!(!Arc::ptr_eq(&first, &recovered));
        let panicked = std::panic::catch_unwind(|| {
            let _: Result<()> = with_session(|_| panic!("callback panic"));
        });
        assert!(panicked.is_err());
        with_session(|contexts| {
            assert!(!Arc::ptr_eq(&recovered, &contexts.cuda));
            Ok(())
        })?;
        clear();
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA and OptiX; run serially outside rendering benchmarks"]
    fn reentrant_and_thread_local_ownership() -> Result<()> {
        clear();
        let outer = with_session(|contexts| {
            let inner = with_session(|inner| Ok(inner.cuda.clone()))?;
            assert!(!Arc::ptr_eq(&inner, &contexts.cuda));
            Ok(contexts.cuda.clone())
        })?;
        // No work is concurrent: join the child before using the parent again.
        let child = std::thread::spawn(|| with_session(|contexts| Ok(contexts.cuda.clone())))
            .join()
            .unwrap()?;
        assert!(!Arc::ptr_eq(&outer, &child));
        with_session(|contexts| {
            assert!(Arc::ptr_eq(&outer, &contexts.cuda));
            Ok(())
        })?;
        clear();
        Ok(())
    }
}
