//! Shared reporting interface for shader preparation and rendering.

use anyhow::Result;

/// Progress, cooperative cancellation, and optional progressive image updates.
pub trait Progress {
    fn log(&mut self, msg: &str);
    fn cancelled(&mut self) -> bool {
        false
    }
    fn wants_preview(&self) -> bool {
        false
    }
    fn preview(
        &mut self,
        _pixels: Vec<f32>,
        _width: u32,
        _height: u32,
        _samples: u32,
        _total: u32,
    ) -> Result<()> {
        Ok(())
    }
}

/// Log to stdout without cancellation or intermediate image readback.
pub struct StdoutProgress;
impl Progress for StdoutProgress {
    fn log(&mut self, msg: &str) {
        println!("{msg}");
    }
}
