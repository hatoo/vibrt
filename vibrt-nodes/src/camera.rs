//! Cycles film filter tables, prepared once on the CPU for camera ray sampling.
pub const FILTER_TABLE_SIZE: usize = 1024;

unsafe extern "C" {
    fn vibrt_filter_table(kind: i32, width: f32, output: *mut f32);
}

pub fn filter_table(kind: u32, width: f32) -> [f32; FILTER_TABLE_SIZE] {
    let mut table = [0.0; FILTER_TABLE_SIZE];
    let width = if width.is_finite() {
        width.clamp(0.01, 100.0)
    } else {
        1.0
    };
    unsafe {
        vibrt_filter_table(kind.min(2) as i32, width, table.as_mut_ptr());
    }
    table
}
