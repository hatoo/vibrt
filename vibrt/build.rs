//! Generate scene-independent BRDF lookup tables.

use std::io::Write;

fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    let path =
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("ggx_energy_lut.rs");
    let mut output = std::io::BufWriter::new(std::fs::File::create(path)?);
    let (e, e_avg, b) = generate_ggx_energy_lut();
    writeln!(output, "pub const GGX_LUT_SIZE: usize = {GGX_LUT_SIZE};")?;
    for (name, values) in [
        ("GGX_ENERGY_LUT", e),
        ("GGX_ENERGY_AVG_LUT", e_avg),
        ("GGX_FRESNEL_LUT", b),
    ] {
        writeln!(output, "pub static {name}: [f32; {}] = [", values.len())?;
        for value in values {
            // Preserve every float bit without decimal formatting/rounding.
            writeln!(output, "    f32::from_bits(0x{:08x}),", value.to_bits())?;
        }
        writeln!(output, "];")?;
    }
    output.flush()
}

const GGX_LUT_SIZE: usize = 32;

/// Monte-Carlo precompute of GGX directional albedo E(cosθ, α) and E_avg(α).
fn generate_ggx_energy_lut() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = GGX_LUT_SIZE;
    let num_samples = 16384u32;
    let mut e_lut = vec![0.0f32; n * n];
    let mut b_lut = vec![0.0f32; n * n];
    let mut e_avg = vec![0.0f32; n];
    for ai in 0..n {
        let alpha = (ai as f32 + 0.5) / n as f32;
        let a2 = alpha * alpha;
        for ci in 0..n {
            let cos_theta = (ci as f32 + 0.5) / n as f32;
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let mut sum = 0.0f64;
            let mut b_sum = 0.0f64;
            for s in 0..num_samples {
                let u1 = s as f64 / num_samples as f64;
                let mut bits = s;
                bits = bits.rotate_right(16);
                bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
                bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
                bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
                bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
                let u2 = bits as f64 / 0x1_0000_0000u64 as f64;
                let a2_d = a2 as f64;
                let cos_h = ((1.0 - u2) / (1.0 + (a2_d - 1.0) * u2)).sqrt();
                let sin_h = (1.0 - cos_h * cos_h).max(0.0).sqrt();
                let phi = 2.0 * std::f64::consts::PI * u1;
                let hx = sin_h * phi.cos();
                let hy = sin_h * phi.sin();
                let hz = cos_h;
                let vx = sin_theta as f64;
                let vy = 0.0;
                let vz = cos_theta as f64;
                let v_dot_h = vx * hx + vy * hy + vz * hz;
                if v_dot_h <= 0.0 {
                    continue;
                }
                let lz = 2.0 * v_dot_h * hz - vz;
                if lz <= 0.0 {
                    continue;
                }
                let n_dot_v = cos_theta as f64;
                let n_dot_l = lz;
                let n_dot_h = cos_h;
                let g1_v =
                    2.0 * n_dot_v / (n_dot_v + (a2_d + (1.0 - a2_d) * n_dot_v * n_dot_v).sqrt());
                let g1_l =
                    2.0 * n_dot_l / (n_dot_l + (a2_d + (1.0 - a2_d) * n_dot_l * n_dot_l).sqrt());
                let weight = g1_v * g1_l * v_dot_h / (n_dot_v * n_dot_h);
                sum += weight;
                let fc = (1.0 - v_dot_h).powi(5);
                b_sum += fc * weight;
            }
            e_lut[ci * n + ai] = (sum / num_samples as f64) as f32;
            b_lut[ci * n + ai] = (b_sum / num_samples as f64) as f32;
        }
        let mut avg = 0.0f64;
        for ci in 0..n {
            let mu = (ci as f64 + 0.5) / n as f64;
            avg += e_lut[ci * n + ai] as f64 * 2.0 * mu / n as f64;
        }
        e_avg[ai] = avg as f32;
    }
    (e_lut, e_avg, b_lut)
}
