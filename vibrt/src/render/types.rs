//! Host-side render options and owned output buffers.
use crate::scene_format;
use anyhow::Result;

/// Per-render overrides supplied by the caller. `None` fields use the scene's
/// render settings, except for the preview cadence documented below.
#[derive(Default, Clone, Copy)]
pub struct RenderOptions {
    pub spp: Option<u32>,
    pub max_depth: Option<u32>,
    pub clamp_indirect: Option<f32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub denoise: bool,
    /// Preview readback cadence; first and last samples are always published.
    /// None uses 100 ms. Zero publishes every completed sample.
    pub preview_interval_ms: Option<u64>,
}

/// Final color, depth, and named AOV buffers with their image dimensions.
pub struct RenderOutput {
    pub aovs: std::collections::BTreeMap<String, Vec<f32>>,
    /// width × height × 4 floats, bottom-left origin, matching Blender's
    /// RenderResult pixel order. RGB is linear; the fourth channel is alpha.
    pub pixels: Vec<f32>,
    /// width × height single-channel primary-ray hit distance from the
    /// camera, in metres. Misses store `cam_clip_end`. Drives the addon's
    /// Mist / Z passes so Cycles-authored compositors (mist haze,
    /// depth-driven masks) work on top of vibrt's output.
    pub depth: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl RenderOptions {
    pub(super) fn settings(
        &self,
        settings: scene_format::RenderSettings,
    ) -> Result<scene_format::RenderSettings> {
        // Apply overrides into a local copy of the render settings. The caller's
        // LoadedScene is borrowed `&` so we can't mutate `settings`; the
        // copy keeps the override semantics that the old `main` had.
        let mut render_settings = settings;
        if let Some(v) = self.spp {
            render_settings.spp = v;
        }
        if let Some(v) = self.max_depth {
            render_settings.max_depth = v;
        }
        if let Some(v) = self.clamp_indirect {
            render_settings.clamp_indirect = v;
        }
        if let Some(v) = self.width {
            render_settings.width = v;
        }
        if let Some(v) = self.height {
            render_settings.height = v;
        }
        render_settings.max_depth = render_settings.max_depth.min(31);
        anyhow::ensure!(render_settings.spp > 0, "samples must be positive");
        anyhow::ensure!(
            render_settings.width > 0 && render_settings.height > 0,
            "image dimensions must be positive"
        );
        Ok(render_settings)
    }
}
