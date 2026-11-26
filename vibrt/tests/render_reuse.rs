//! CUDA/OptiX integration test for session, scene and job ownership.
use anyhow::Result;
use vibrt_native::{
    load_scene_from_bytes,
    render::{GpuScene, RenderJob, RenderOptions, RenderSession},
    Progress, StdoutProgress,
};

fn upload(session: &RenderSession, aovs: bool) -> Result<GpuScene<'_>> {
    // All source data is local: returning GpuScene proves it borrows only session.
    let vertices = [-3.0f32, -3.0, 0.0, 3.0, -3.0, 0.0, 0.0, 3.0, 0.0];
    let indices = [0u32, 1, 2];
    let identity = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    ];
    let mut camera = identity;
    camera[11] = 3.0;
    let description = serde_json::json!({
        "version": 1,
        "render": {"width": 24, "height": 16, "spp": 8, "max_depth": 4},
        "camera": {"transform": camera, "fov_y_rad": 0.8},
        "meshes": [{"vertices": 0, "indices": 1}],
        "materials": [{"base_color": [0.8, 0.2, 0.1], "emission": [1.0, 0.25, 0.125]}],
        "objects": [{"mesh": 0, "material": 0, "transform": identity}],
        "aov_names": if aovs { vec!["unused"] } else { vec![] },
        "world": null
    });
    let loaded = load_scene_from_bytes(
        &description.to_string(),
        &[
            bytemuck::cast_slice(&vertices),
            bytemuck::cast_slice(&indices),
        ],
        &[],
    )?;
    GpuScene::upload(session, &loaded, &mut StdoutProgress)
}

struct FailingPreview(bool);
impl Progress for FailingPreview {
    fn log(&mut self, _: &str) {}
    fn wants_preview(&self) -> bool {
        true
    }
    fn preview(&mut self, _: Vec<f32>, _: u32, _: u32, _: u32, _: u32) -> Result<()> {
        if self.0 {
            panic!("callback panic");
        }
        anyhow::bail!("callback error")
    }
}

#[test]
#[ignore = "requires CUDA and OptiX; run serially outside rendering benchmarks"]
fn uploaded_scene_survives_sources_variants_and_failed_jobs() -> Result<()> {
    let session = RenderSession::new()?;
    let scene = upload(&session, false)?;
    let options = RenderOptions::default();
    let reference = RenderJob::new(&scene, options).render(&mut StdoutProgress)?;
    assert!(reference.pixels.iter().any(|&v| v > 0.0));

    // Evict the session's single cached shader variant while scene still uses it.
    let other = upload(&session, true)?;
    let aovs = RenderJob::new(&other, options).render(&mut StdoutProgress)?;
    assert_eq!(aovs.aovs["unused"].len(), 24 * 16 * 4);
    drop(other);

    let resized = RenderJob::new(
        &scene,
        RenderOptions {
            width: Some(12),
            height: Some(8),
            spp: Some(1),
            denoise: true,
            ..options
        },
    )
    .render(&mut StdoutProgress)?;
    assert_eq!((resized.width, resized.height), (12, 8));
    assert_eq!(resized.pixels.len(), 12 * 8 * 4);
    assert!(resized.pixels.iter().all(|v| v.is_finite()));

    assert!(RenderJob::new(&scene, options)
        .render(&mut FailingPreview(false))
        .is_err());
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = RenderJob::new(&scene, options).render(&mut FailingPreview(true));
    }));
    assert!(panicked.is_err());

    let repeated = RenderJob::new(&scene, options).render(&mut StdoutProgress)?;
    assert_eq!(repeated.pixels, reference.pixels);
    assert_eq!(repeated.depth, reference.depth);
    Ok(())
}
