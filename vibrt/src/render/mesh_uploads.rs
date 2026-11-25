//! Bounded pinned mesh uploads that overlap OptiX acceleration builds.

use crate::CudaResultExt;
use anyhow::Result;
use cudarc::driver::{
    sys::CUevent_flags, CudaEvent, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice,
};
use std::mem;
use std::sync::Arc;

const STAGING_BYTES: usize = 1 << 20;
const STAGING_PAGES: usize = 4;

struct Page {
    _storage: PinnedHostSlice<u8>,
    pointer: *mut u8,
    complete: CudaEvent,
}

/// Own device arrays through rendering and append small uploads to disjoint
/// regions of each host page. Only recycling a full page waits for its DMA.
pub(super) struct MeshUploads {
    stream: Arc<CudaStream>,
    consumer: Arc<CudaStream>,
    ready: CudaEvent,
    buffers: Vec<CudaSlice<u8>>,
    pages: Vec<Page>,
    next: usize,
    used: usize,
}

impl MeshUploads {
    pub(super) fn new(consumer: Arc<CudaStream>) -> Result<Self> {
        let stream = consumer.context().new_stream().cuda()?;
        Self::with_streams(stream, consumer)
    }

    fn with_streams(stream: Arc<CudaStream>, consumer: Arc<CudaStream>) -> Result<Self> {
        let ready = stream
            .context()
            .new_event(Some(CUevent_flags::CU_EVENT_DISABLE_TIMING))
            .cuda()?;
        Ok(Self {
            stream,
            consumer,
            ready,
            buffers: Vec::new(),
            pages: Vec::new(),
            next: 0,
            used: 0,
        })
    }

    pub(super) fn copy_slice<T>(&mut self, data: &[T]) -> Result<u64> {
        let size = mem::size_of_val(data);
        if size == 0 {
            return Ok(0);
        }
        // Callers supply primitive arrays or padding-free repr(C) QueryNodes,
        // matching the renderer's raw ABI uploads.
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size) };
        let mut buffer = unsafe { self.stream.alloc::<u8>(size) }.cuda()?;
        let mut offset = 0;
        while offset < size {
            if self.next == self.pages.len() {
                let mut storage =
                    unsafe { self.stream.context().alloc_pinned(STAGING_BYTES) }.cuda()?;
                let pointer = storage.as_mut_ptr().cuda()?;
                let complete = self
                    .stream
                    .context()
                    .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                    .cuda()?;
                self.pages.push(Page {
                    _storage: storage,
                    pointer,
                    complete,
                });
            } else if self.used == 0 {
                // This event follows the last copy from the full page. A
                // partial page remains live until finish() or Drop drains it.
                self.pages[self.next].complete.synchronize().cuda()?;
            }
            let page = &self.pages[self.next];
            let count = (STAGING_BYTES - self.used).min(size - offset);
            // Earlier DMA reads only the used prefix. Initialize a disjoint
            // region without creating a mutable reference to that prefix.
            let chunk = unsafe {
                let destination = page.pointer.add(self.used);
                std::ptr::copy_nonoverlapping(bytes.as_ptr().add(offset), destination, count);
                std::slice::from_raw_parts(destination, count)
            };
            self.stream
                .memcpy_htod(chunk, &mut buffer.slice_mut(offset..offset + count))
                .cuda()?;
            self.used += count;
            offset += count;
            if self.used == STAGING_BYTES {
                page.complete.record(&self.stream).cuda()?;
                self.next = (self.next + 1) % STAGING_PAGES;
                self.used = 0;
            }
        }
        let (pointer, record) = buffer.device_ptr(&self.stream);
        drop(record);
        self.buffers.push(buffer);
        Ok(pointer)
    }

    /// Order a raw OptiX consumer after all uploads queued so far. CUDA
    /// captures this recording when the wait is enqueued; later recordings
    /// of the event do not change an earlier wait.
    pub(super) fn wait_on_consumer(&self) -> Result<()> {
        self.ready.record(&self.stream).cuda()?;
        self.consumer.wait(&self.ready).cuda()?;
        Ok(())
    }

    /// Release pinned storage after the existing mesh-preparation barrier;
    /// device arrays stay owned until rendering completes.
    pub(super) fn finish(&mut self) -> Result<()> {
        self.stream.synchronize().cuda()?;
        self.consumer.synchronize().cuda()?;
        self.pages.clear();
        self.next = 0;
        self.used = 0;
        Ok(())
    }
}

impl Drop for MeshUploads {
    fn drop(&mut self) {
        // Raw OptiX consumers are not tracked by CudaSlice. Drain both
        // streams before releasing host pages or device arrays, even on an
        // early return after finish() but before rendering has completed.
        self.stream.context().record_err(self.stream.synchronize());
        self.consumer
            .context()
            .record_err(self.consumer.synchronize());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{result, CudaContext, DevicePtrMut};

    #[test]
    #[ignore = "requires a CUDA GPU; run serially outside rendering benchmarks"]
    fn staging_wrap_and_chunk_tails_preserve_device_bytes() -> Result<()> {
        let stream = CudaContext::new(0).cuda()?.default_stream();
        let mut uploads = MeshUploads::new(stream.clone())?;
        assert_eq!(uploads.copy_slice::<u8>(&[])?, 0);
        assert!(uploads.pages.is_empty());
        let mut expected = Vec::new();
        for index in 0..24 {
            let size = match index % 6 {
                0 => 31,
                1 => 17,
                2 => STAGING_BYTES - 1,
                3 => STAGING_BYTES + 1,
                4 => STAGING_BYTES * (STAGING_PAGES + 1) + 17,
                _ => 3,
            };
            let data: Vec<u8> = (0..size).map(|i| (i * 7 + index * 31) as u8).collect();
            assert_ne!(uploads.copy_slice(&data)?, 0);
            if index == 1 {
                assert_eq!(uploads.pages.len(), 1);
                assert_eq!(uploads.used, 48);
            }
            expected.push(data);
        }
        assert_eq!(uploads.pages.len(), STAGING_PAGES);
        uploads.finish()?;
        assert!(uploads.pages.is_empty());
        for (buffer, expected) in uploads.buffers.iter().zip(&expected) {
            assert_eq!(stream.clone_dtoh(buffer).cuda()?, *expected);
        }
        // Reuse after finish, then drop with an unfinished partial page. Keep
        // the destinations alive to verify that Drop drained their DMA reads.
        uploads.buffers.clear();
        for data in &expected[..4] {
            uploads.copy_slice(data)?;
        }
        let buffers = mem::take(&mut uploads.buffers);
        drop(uploads);
        for (buffer, expected) in buffers.iter().zip(&expected) {
            assert_eq!(stream.clone_dtoh(buffer).cuda()?, *expected);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA GPU; run serially outside rendering benchmarks"]
    fn raw_consumers_wait_for_uploads_and_finish_before_drop() -> Result<()> {
        let stream = CudaContext::new(0).cuda()?.default_stream();
        let mut uploads = MeshUploads::new(stream.clone())?;
        let mut copies = Vec::new();
        for index in 0..32 {
            let size = STAGING_BYTES / 2 + index * 17;
            let expected: Vec<u8> = (0..size).map(|i| (i * 7 + index * 31) as u8).collect();
            let source = uploads.copy_slice(&expected)?;
            let mut copy = unsafe { stream.alloc::<u8>(size) }.cuda()?;
            uploads.wait_on_consumer()?;
            // Deliberately bypass CudaSlice's source dependency tracking, as
            // OptiX does when consuming the integer vertex/index addresses.
            {
                let (destination, _record) = copy.device_ptr_mut(&stream);
                unsafe { result::memcpy_dtod_async(destination, source, size, stream.cu_stream()) }
                    .cuda()?;
            }
            copies.push((copy, expected));
        }
        // No explicit finish: exercise the error/early-return cleanup path
        // with queued raw consumers and partially filled pinned pages.
        drop(uploads);
        for (copy, expected) in copies {
            assert_eq!(stream.clone_dtoh(&copy).cuda()?, expected);
        }
        Ok(())
    }
}
