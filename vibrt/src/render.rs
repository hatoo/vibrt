//! Rendering facade. Sessions own contexts; scenes own uploads; jobs own images.
mod gpu_scene;
mod job;
mod lighting;
mod mesh_uploads;
mod query_cache;
mod sample_batches;
mod session;
mod types;
mod upload;
pub use crate::progress::{Progress, StdoutProgress};
use crate::scene_loader::LoadedScene;
use anyhow::Result;
pub use gpu_scene::GpuScene;
pub use job::RenderJob;
pub use session::RenderSession;
pub use types::{RenderOptions, RenderOutput};
/// Render `scene` to in-memory color, depth, and AOV buffers.
///
/// Applies `opts` without mutating the caller's scene. Each launch completes
/// one sample, including when previews are disabled; optional denoising runs
/// after sampling, before the final readback.
pub fn render_to_pixels(
    scene: &LoadedScene,
    opts: &RenderOptions,
    progress: &mut dyn Progress,
) -> Result<RenderOutput> {
    let render_settings = opts.settings(scene.file.render)?;
    anyhow::ensure!(!progress.cancelled(), "render cancelled");

    progress.log(&format!(
        "Scene: {}x{} @ {} spp, {} objects, {} meshes, {} mats, {} textures, lights(p/s/spot/rect/mesh): {}/{}/{}/{}/{}",
        render_settings.width,
        render_settings.height,
        render_settings.spp,
        scene.objects.len(),
        scene.meshes.len(),
        scene.file.materials.len(),
        scene.textures.len(),
        scene.point_lights.len(),
        scene.sun_lights.len(),
        scene.spot_lights.len(),
        scene.rect_lights.len(),
        scene.mesh_lights.len(),
    ));

    let t_prepare = std::time::Instant::now();
    session::with_session(|session| {
        let context_time = t_prepare.elapsed();
        let gpu_scene = GpuScene::upload(session, scene, progress)?;
        RenderJob::new(&gpu_scene, *opts).run(progress, Some((t_prepare, context_time)))
    })
}
