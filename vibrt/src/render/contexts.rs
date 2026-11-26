//! Reuse owning GPU contexts and bounded CPU query data on one render thread.

use crate::CudaResultExt;
use anyhow::{Context, Result};
use cudarc::driver::CudaContext;
use optix::{DeviceContext, DeviceContextOptions};
use std::cell::RefCell;
use std::sync::Arc;

// Fields drop in declaration order: OptiX must go before its CUDA context.
pub(super) struct Contexts {
    pub optix: DeviceContext,
    pub cuda: Arc<CudaContext>,
    pub queries: RefCell<super::query_cache::QueryCache>,
}

impl Contexts {
    fn new() -> Result<Self> {
        let cuda = CudaContext::new(0).cuda().context("CUDA context")?;
        let handle = optix::init().context("OptiX init")?;
        let optix = DeviceContext::new(
            &handle,
            cuda.cu_ctx() as optix_sys::CUcontext,
            &DeviceContextOptions::default(),
        )
        .context("OptiX context")?;
        Ok(Self {
            optix,
            cuda,
            queries: RefCell::new(super::query_cache::QueryCache::default()),
        })
    }
}

impl Drop for Contexts {
    fn drop(&mut self) {
        // A Blender callback or another CUDA user may have switched contexts.
        self.cuda.record_err(self.cuda.bind_to_thread());
    }
}

thread_local! {
    // Only owned CPU query data accompanies the contexts: no device scene
    // buffers, pipelines, RNA pointers or callbacks survive a render.
    // Each thread owns at most one idle pair, released when the thread exits.
    static IDLE: RefCell<Option<Contexts>> = const { RefCell::new(None) };
}

pub(super) fn with_contexts<T>(render: impl FnOnce(&Contexts) -> Result<T>) -> Result<T> {
    // Take the entry out while rendering. Reentrant callbacks can acquire their
    // own pair without borrowing the cache or sharing an active OptiX context.
    let contexts = match IDLE.with(|idle| idle.borrow_mut().take()) {
        Some(contexts) => contexts,
        None => Contexts::new()?,
    };
    contexts.cuda.bind_to_thread().cuda()?;
    let result = render(&contexts);
    // The render's local resources have dropped, including queued async frees.
    // Observe cleanup errors and drain work before caching or destroying OptiX.
    let drained = contexts.cuda.synchronize().cuda();
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
        let first = with_contexts(|contexts| Ok(contexts.cuda.clone()))?;
        // Simulate another CUDA user switching away between render callbacks.
        let other = CudaContext::new_non_primary(0, 0).cuda()?;
        with_contexts(|contexts| {
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
        let failed: Result<()> = with_contexts(|_| anyhow::bail!("cancelled callback"));
        assert!(failed.is_err());
        let recovered = with_contexts(|contexts| Ok(contexts.cuda.clone()))?;
        assert!(!Arc::ptr_eq(&first, &recovered));
        let panicked = std::panic::catch_unwind(|| {
            let _: Result<()> = with_contexts(|_| panic!("callback panic"));
        });
        assert!(panicked.is_err());
        with_contexts(|contexts| {
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
        let outer = with_contexts(|contexts| {
            let inner = with_contexts(|inner| Ok(inner.cuda.clone()))?;
            assert!(!Arc::ptr_eq(&inner, &contexts.cuda));
            Ok(contexts.cuda.clone())
        })?;
        // No work is concurrent: join the child before using the parent again.
        let child = std::thread::spawn(|| with_contexts(|contexts| Ok(contexts.cuda.clone())))
            .join()
            .unwrap()?;
        assert!(!Arc::ptr_eq(&outer, &child));
        with_contexts(|contexts| {
            assert!(Arc::ptr_eq(&outer, &contexts.cuda));
            Ok(())
        })?;
        clear();
        Ok(())
    }
}
