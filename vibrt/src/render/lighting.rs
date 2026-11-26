//! Scene-independent lighting distributions and BRDF lookup tables.

// The integration depends only on the BRDF, never on the scene or GPU.
// Build it once and embed the exact float bits, including for the first render.
include!(concat!(env!("OUT_DIR"), "/ggx_energy_lut.rs"));

/// Build marginal + conditional CDFs for envmap importance sampling (RGB data).
pub fn build_envmap_cdf(rgb: &[f32], width: u32, height: u32) -> (Vec<f32>, Vec<f32>, f32) {
    let w = width as usize;
    let h = height as usize;
    let mut cond = vec![0.0f32; h * (w + 1)];
    let mut row_int = vec![0.0f32; h];
    for y in 0..h {
        let sin_t = (std::f32::consts::PI * (y as f32 + 0.5) / h as f32).sin();
        let ro = y * (w + 1);
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let lum = 0.2126 * rgb[idx] + 0.7152 * rgb[idx + 1] + 0.0722 * rgb[idx + 2];
            cond[ro + x + 1] = cond[ro + x] + lum * sin_t;
        }
        row_int[y] = cond[ro + w];
        if row_int[y] > 0.0 {
            let inv = 1.0 / row_int[y];
            for x in 1..=w {
                cond[ro + x] *= inv;
            }
        }
    }
    let mut marg = vec![0.0f32; h + 1];
    for y in 0..h {
        marg[y + 1] = marg[y] + row_int[y];
    }
    let total = marg[h];
    if total > 0.0 {
        let inv = 1.0 / total;
        for y in 1..=h {
            marg[y] *= inv;
        }
    }
    (marg, cond, total)
}

// Verify the generated table dimensions at compilation.
const _: () = assert!(GGX_ENERGY_LUT.len() == GGX_LUT_SIZE * GGX_LUT_SIZE);
