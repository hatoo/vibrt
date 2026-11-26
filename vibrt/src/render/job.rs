//! Per-render image buffers, sample scheduling, denoising and readback.
use super::{
    gpu_scene::GpuScene,
    sample_batches::{SampleBatches, MAX_BATCH_SAMPLES},
    types::{RenderOptions, RenderOutput},
    upload::*,
};
use crate::{gpu_types::LaunchParams, CudaResultExt, Progress};
use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, DevicePtr};
use optix::{denoiser, PixelFormat};
use std::{mem, time::Instant};

/// A sampling job borrows an uploaded scene; all image buffers are job-local.
pub struct RenderJob<'scene, 'session> {
    scene: &'scene GpuScene<'session>,
    options: RenderOptions,
}

impl<'scene, 'session> RenderJob<'scene, 'session> {
    pub fn new(scene: &'scene GpuScene<'session>, options: RenderOptions) -> Self {
        Self { scene, options }
    }

    pub fn render(self, progress: &mut dyn Progress) -> Result<RenderOutput> {
        self.run(progress, None)
    }

    pub(super) fn run(
        self,
        progress: &mut dyn Progress,
        preparation: Option<(Instant, std::time::Duration)>,
    ) -> Result<RenderOutput> {
        let scene = self.scene;
        let opts = &self.options;
        let render_settings = opts.settings(scene.settings)?;
        anyhow::ensure!(!progress.cancelled(), "render cancelled");
        let stream = scene.stream()?;
        let cu_stream = stream.cu_stream() as optix_sys::CUstream;
        let t_buffers = Instant::now();
        let (t_prepare, context_time, ptx_time, pipeline_time, upload_time) = match preparation {
            Some((start, context)) => (
                start,
                context,
                scene.ptx_time,
                scene.pipeline_time,
                scene.upload_time,
            ),
            None => (
                t_buffers,
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
            ),
        };
        // --- Image buffer (float4 per pixel) ---
        let pixel_count = (render_settings.width * render_settings.height) as usize;
        let d_image: CudaSlice<f32> = stream.alloc_zeros(pixel_count * 4).cuda()?;
        let d_aovs = if scene.aov_count() == 0 {
            None
        } else {
            Some(
                stream
                    .alloc_zeros::<f32>(pixel_count * 4 * scene.aov_count())
                    .cuda()?,
            )
        };
        let progressive = progress.wants_preview();
        // Keep one sample per launch: a long per-pixel loop leaves the GPU waiting
        // for the slowest paths. Queue short groups of launches between CPU waits.
        let d_accumulation: Option<CudaSlice<f32>> = if render_settings.spp > 1 {
            Some(stream.alloc_zeros(pixel_count * 4).cuda()?)
        } else {
            None
        };

        // Denoiser guide AOVs: allocated (float3 per pixel) only when --denoise.
        let d_albedo: Option<CudaSlice<f32>> = if opts.denoise {
            Some(stream.alloc_zeros(pixel_count * 3).cuda()?)
        } else {
            None
        };
        let d_normal: Option<CudaSlice<f32>> = if opts.denoise {
            Some(stream.alloc_zeros(pixel_count * 3).cuda()?)
        } else {
            None
        };
        let albedo_aov = d_albedo.as_ref().map_or(0, |s| dptr_f32(s, &stream));
        let normal_aov = d_normal.as_ref().map_or(0, |s| dptr_f32(s, &stream));

        // Per-pixel primary-ray hit distance. Always allocated; consumed by
        // the addon to populate Mist / Z passes for the compositor.
        let d_depth: CudaSlice<f32> = stream.alloc_zeros(pixel_count).cuda()?;
        let depth_aov = dptr_f32(&d_depth, &stream);

        // Single u32 counter for the camera-ray back-face skip's exhausted-cap
        // event (see trace_path in devicecode.cu). Always allocated so the
        // kernel never has to null-check the pointer; the host warns after
        // the render if it came back non-zero.
        let d_back_face_warn: CudaSlice<u32> = stream.alloc_zeros(1).cuda()?;
        let primary_back_face_skip_exhausted = dptr_u32(&d_back_face_warn, &stream);

        let mut lp = scene.launch_params(render_settings);
        lp.aov_data = d_aovs.as_ref().map_or(0, |s| dptr_f32(s, &stream));
        lp.image = dptr_f32(&d_image, &stream);
        lp.accumulation = d_accumulation.as_ref().map_or(0, |s| dptr_f32(s, &stream));
        lp.albedo_aov = albedo_aov;
        lp.normal_aov = normal_aov;
        lp.depth_aov = depth_aov;
        lp.primary_back_face_skip_exhausted = primary_back_face_skip_exhausted;
        // Distinct parameter slots let consecutive launches run without a host
        // upload/wait between samples. Reuse the slots only after the batch ends.
        let mut batch_params = vec![lp; MAX_BATCH_SAMPLES.min(render_settings.spp as usize)];
        let mut d_params = stream.clone_htod(&batch_params).cuda()?;
        let batch_start = stream
            .context()
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .cuda()?;
        let batch_end = stream
            .context()
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .cuda()?;
        stream.synchronize().cuda()?;
        progress.log(&format!(
        "GPU preparation: {:.2?} (context={:.2?}, PTX={:.2?}, pipeline={:.2?}, scene={:.2?}, buffers={:.2?})",
        t_prepare.elapsed(), context_time, ptx_time, pipeline_time, upload_time,
        t_buffers.elapsed(),
    ));

        progress.log("Rendering...");
        let t0 = std::time::Instant::now();
        let interval = std::time::Duration::from_millis(opts.preview_interval_ms.unwrap_or(100));
        let every_sample = progressive && interval.is_zero();
        let mut batches = SampleBatches::new(progressive.then_some(interval));
        let mut last_preview = None;
        let mut queue_batch = |start: u32, count: u32| -> Result<()> {
            for (i, params) in batch_params[..count as usize].iter_mut().enumerate() {
                params.sample_start = start + i as u32;
            }
            stream
                .memcpy_htod(&batch_params[..count as usize], &mut d_params)
                .cuda()?;
            if !every_sample {
                batch_start.record(&stream).cuda()?;
            }
            let (params_ptr, _record_params) = d_params.device_ptr(&stream);
            for i in 0..count {
                // Every parameter slot and output allocation outlives the drain below.
                unsafe {
                    scene.launch(
                        &stream,
                        params_ptr + i as u64 * mem::size_of::<LaunchParams>() as u64,
                        render_settings.width,
                        render_settings.height,
                    )?;
                }
            }
            if !every_sample {
                batch_end.record(&stream).cuda()?;
            }
            Ok(())
        };
        let pending_work = PendingWork(Some(&stream));
        let sampling = (|| -> Result<()> {
            let mut completed = 0;
            let mut pending = 0;
            while completed < render_settings.spp {
                anyhow::ensure!(!progress.cancelled(), "render cancelled");
                if pending == 0 {
                    pending = batches.next_count(render_settings.spp - completed);
                    queue_batch(completed, pending)?;
                }
                stream.synchronize().cuda()?;
                if !every_sample {
                    batches.record_gpu_time(pending, batch_start.elapsed_ms(&batch_end).cuda()?);
                }
                completed += pending;
                pending = 0;
                anyhow::ensure!(!progress.cancelled(), "render cancelled");
                if progressive
                    && (completed == render_settings.spp
                        || last_preview.is_none_or(|t: std::time::Instant| t.elapsed() >= interval))
                {
                    let pixels = stream.clone_dtoh(&d_image).cuda()?;
                    if !every_sample && completed < render_settings.spp {
                        // The snapshot owns its host memory. Let the GPU work on
                        // the next batch while Blender displays this completed one.
                        pending = batches.next_count(render_settings.spp - completed);
                        queue_batch(completed, pending)?;
                    }
                    progress.preview(
                        pixels,
                        render_settings.width,
                        render_settings.height,
                        completed,
                        render_settings.spp,
                    )?;
                    last_preview = Some(std::time::Instant::now());
                }
            }
            anyhow::ensure!(!progress.cancelled(), "render cancelled");
            Ok(())
        })();
        // A callback/cancel/launch error may leave the next batch in flight. Drain
        // it before dropping parameters, scene buffers, SBT or the OptiX pipeline.
        let drained = pending_work.finish();
        sampling?;
        drained?;
        progress.log(&format!("Render: {:.2?}", t0.elapsed()));

        // Read back the camera-ray back-face skip counter. Non-zero means the
        // camera ray hit > 4 back-facing surfaces in a row on at least one
        // sample — clip_start was set inside an unusually deep stack of
        // hidden geometry, and the affected pixels fell back to shading the
        // last back-face we hit. Emit one warn line so this isn't silent.
        let back_face_warn: Vec<u32> = stream.clone_dtoh(&d_back_face_warn).cuda()?;
        if let Some(&n) = back_face_warn.first() {
            if n > 0 {
                let spp = render_settings.spp.max(1) as f64;
                let est_pixels = (n as f64 / spp).round() as u64;
                progress.log(&format!(
                    "[vibrt] warn: camera-ray back-face skip exhausted its retry \
                 cap on {n} samples (~{est_pixels} pixels at {spp} spp); \
                 those pixels fall back to shading the last back-face. \
                 Likely cause: clip_start lands the camera inside thick / \
                 deeply-nested geometry."
                ));
            }
        }

        // --- Optional OptiX denoiser (AOV model, HDR float4, no guides) ---
        let d_final = if opts.denoise {
            progress.log("Denoising...");
            let t_dn = std::time::Instant::now();
            let w = render_settings.width;
            let h = render_settings.height;

            let denoiser = scene.denoiser()?;

            let sizes = denoiser
                .compute_memory_resources(w, h)
                .context("denoiser sizes")?;
            let d_state: CudaSlice<u8> = unsafe { stream.alloc(sizes.state_size) }.cuda()?;
            let d_scratch: CudaSlice<u8> =
                unsafe { stream.alloc(sizes.without_overlap_scratch_size) }.cuda()?;
            let d_intensity_scratch: CudaSlice<u8> =
                unsafe { stream.alloc(sizes.compute_intensity_size) }.cuda()?;
            let d_intensity: CudaSlice<f32> = stream.alloc_zeros(1).cuda()?;
            let d_denoised: CudaSlice<f32> = stream.alloc_zeros(pixel_count * 4).cuda()?;

            denoiser
                .setup(
                    cu_stream,
                    w,
                    h,
                    dptr_u8(&d_state, &stream),
                    sizes.state_size,
                    dptr_u8(&d_scratch, &stream),
                    sizes.without_overlap_scratch_size,
                )
                .context("denoiser setup")?;

            let pixel_stride = 4 * mem::size_of::<f32>() as u32;
            let input_image = denoiser::Image2D {
                data: dptr_f32(&d_image, &stream),
                width: w,
                height: h,
                row_stride: w * pixel_stride,
                pixel_stride,
                format: PixelFormat::Float4,
            };
            let output_image = denoiser::Image2D {
                data: dptr_f32(&d_denoised, &stream),
                ..input_image
            };

            denoiser
                .compute_intensity(
                    cu_stream,
                    &input_image,
                    dptr_f32(&d_intensity, &stream),
                    dptr_u8(&d_intensity_scratch, &stream),
                    sizes.compute_intensity_size,
                )
                .context("denoiser compute_intensity")?;

            let params = denoiser::DenoiserParams {
                hdr_intensity: dptr_f32(&d_intensity, &stream),
                blend_factor: 0.0,
                hdr_average_color: 0,
                temporal_mode_use_previous_layers: false,
            };
            let layer = denoiser::DenoiserLayer {
                input: input_image,
                output: output_image,
                previous_output: None,
            };

            let f3_stride = 3 * mem::size_of::<f32>() as u32;
            let guide_albedo_img = d_albedo.as_ref().map(|s| denoiser::Image2D {
                data: dptr_f32(s, &stream),
                width: w,
                height: h,
                row_stride: w * f3_stride,
                pixel_stride: f3_stride,
                format: PixelFormat::Float3,
            });
            let guide_normal_img = d_normal.as_ref().map(|s| denoiser::Image2D {
                data: dptr_f32(s, &stream),
                width: w,
                height: h,
                row_stride: w * f3_stride,
                pixel_stride: f3_stride,
                format: PixelFormat::Float3,
            });
            let guide = denoiser::DenoiserGuideLayer {
                albedo: guide_albedo_img,
                normal: guide_normal_img,
                flow: None,
            };

            denoiser
                .invoke(
                    cu_stream,
                    &params,
                    dptr_u8(&d_state, &stream),
                    sizes.state_size,
                    &guide,
                    &[layer],
                    0,
                    0,
                    dptr_u8(&d_scratch, &stream),
                    sizes.without_overlap_scratch_size,
                )
                .context("denoiser invoke")?;
            stream.synchronize().cuda()?;
            progress.log(&format!("Denoise: {:.2?}", t_dn.elapsed()));

            // Keep scratch/state buffers alive until the stream has caught up.
            drop((d_state, d_scratch, d_intensity_scratch, d_intensity));
            d_denoised
        } else {
            d_image
        };

        // --- Readback ---
        let pixels: Vec<f32> = stream.clone_dtoh(&d_final).cuda()?;
        let depth: Vec<f32> = stream.clone_dtoh(&d_depth).cuda()?;
        let mut aovs = std::collections::BTreeMap::new();
        if let Some(data) = &d_aovs {
            let values: Vec<f32> = stream.clone_dtoh(data).cuda()?;
            aovs = scene.collect_aovs(&values, pixel_count, render_settings.spp);
        }

        Ok(RenderOutput {
            aovs,
            pixels,
            depth,
            width: render_settings.width,
            height: render_settings.height,
        })
    }
}

// A Rust progress callback can unwind while the next batch is running. This
// guard drops before job-local buffers, including the launch-parameter slots.
struct PendingWork<'a>(Option<&'a cudarc::driver::CudaStream>);

impl PendingWork<'_> {
    fn finish(mut self) -> Result<()> {
        self.0.take().unwrap().synchronize().cuda()
    }
}

impl Drop for PendingWork<'_> {
    fn drop(&mut self) {
        if let Some(stream) = self.0 {
            let context = stream.context();
            context.record_err(context.bind_to_thread());
            context.record_err(stream.synchronize());
        }
    }
}
