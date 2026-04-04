//! Parse PBRT's tensor_file BSDF format and approximate as analytic materials.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

struct TensorField {
    dtype: u8,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn type_size(dtype: u8) -> usize {
    match dtype {
        1 | 2 => 1, // u8, i8
        3 | 4 => 2, // u16, i16
        5 | 6 => 4, // u32, i32
        7 | 8 => 8, // u64, i64
        9 => 2,     // f16
        10 => 4,    // f32
        11 => 8,    // f64
        _ => 0,
    }
}

fn parse_tensor_file(path: &Path) -> Option<HashMap<String, TensorField>> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut header = [0u8; 12];
    file.read_exact(&mut header).ok()?;
    if &header != b"tensor_file\0" {
        return None;
    }

    let mut version = [0u8; 2];
    file.read_exact(&mut version).ok()?;
    if version != [1, 0] {
        return None;
    }

    let mut n_fields_buf = [0u8; 4];
    file.read_exact(&mut n_fields_buf).ok()?;
    let n_fields = u32::from_le_bytes(n_fields_buf);

    let mut fields = HashMap::new();

    for _ in 0..n_fields {
        let mut name_len_buf = [0u8; 2];
        file.read_exact(&mut name_len_buf).ok()?;
        let name_len = u16::from_le_bytes(name_len_buf) as usize;

        let mut name_buf = vec![0u8; name_len];
        file.read_exact(&mut name_buf).ok()?;
        let name = String::from_utf8_lossy(&name_buf).to_string();

        let mut ndim_buf = [0u8; 2];
        file.read_exact(&mut ndim_buf).ok()?;
        let ndim = u16::from_le_bytes(ndim_buf) as usize;

        let mut dtype_buf = [0u8; 1];
        file.read_exact(&mut dtype_buf).ok()?;
        let dtype = dtype_buf[0];

        let mut offset_buf = [0u8; 8];
        file.read_exact(&mut offset_buf).ok()?;
        let offset = u64::from_le_bytes(offset_buf);

        let mut shape = Vec::with_capacity(ndim);
        let mut total_size = type_size(dtype);
        for _ in 0..ndim {
            let mut dim_buf = [0u8; 8];
            file.read_exact(&mut dim_buf).ok()?;
            let dim = u64::from_le_bytes(dim_buf);
            shape.push(dim);
            total_size *= dim as usize;
        }

        let cur_pos = file.stream_position().ok()?;
        file.seek(SeekFrom::Start(offset)).ok()?;
        let mut data = vec![0u8; total_size];
        file.read_exact(&mut data).ok()?;
        file.seek(SeekFrom::Start(cur_pos)).ok()?;

        fields.insert(name, TensorField { dtype, shape, data });
    }

    Some(fields)
}

fn field_as_f32(field: &TensorField) -> Vec<f32> {
    if field.dtype == 10 {
        // Float32
        field
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else if field.dtype == 11 {
        // Float64
        field
            .data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect()
    } else {
        Vec::new()
    }
}

/// Material approximation extracted from a measured BSDF
pub struct BsdfApprox {
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub is_metallic: bool,
    pub eta: [f32; 3],
    pub k: [f32; 3],
}

/// Load a PBRT tensor_file BSDF and approximate its properties.
pub fn load_and_approximate(path: &Path) -> Option<BsdfApprox> {
    let fields = parse_tensor_file(path)?;

    let wavelengths = fields.get("wavelengths")?;
    let wavelengths_f32 = field_as_f32(wavelengths);
    let n_wavelengths = wavelengths_f32.len();
    if n_wavelengths == 0 {
        return None;
    }

    // Find wavelength indices closest to R=630nm, G=532nm, B=467nm
    let find_idx = |target: f32| -> usize {
        wavelengths_f32
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a - target).abs())
                    .partial_cmp(&((**b - target).abs()))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    let rgb_idx = [find_idx(630.0), find_idx(532.0), find_idx(467.0)];

    // Use the luminance field: shape [n_phi_i, n_theta_i, n_u, n_v]
    // This is the VNDF-weighted luminance of the BSDF.
    // We sample at a few angles to estimate reflectance properties.
    let luminance = fields.get("luminance")?;
    let lum = field_as_f32(luminance);
    if luminance.shape.len() != 4 {
        return None;
    }
    let n_phi = luminance.shape[0] as usize;
    let n_theta = luminance.shape[1] as usize;
    let n_u = luminance.shape[2] as usize;
    let n_v = luminance.shape[3] as usize;
    if n_phi == 0 || n_theta == 0 || n_u == 0 || n_v == 0 {
        return None;
    }

    // Compute overall average luminance (brightness indicator)
    let total = lum.len();
    let avg_lum: f32 = if total > 0 {
        lum.iter().sum::<f32>() / total as f32
    } else {
        0.0
    };

    // Sample luminance at normal incidence (theta=0, phi=0) center of VNDF
    let center_u = n_u / 2;
    let center_v = n_v / 2;
    let normal_lum = lum[0 * n_theta * n_u * n_v + 0 * n_u * n_v + center_u * n_v + center_v];

    // Sample luminance at grazing (last theta)
    let _grazing_lum =
        lum[0 * n_theta * n_u * n_v + (n_theta - 1) * n_u * n_v + center_u * n_v + center_v];

    // Use spectra to get color: shape [n_phi, n_theta, n_wavelengths, n_u, n_v]
    let spectra = fields.get("spectra")?;
    let spec = field_as_f32(spectra);
    if spectra.shape.len() != 5 {
        return None;
    }
    let sp = &spectra.shape;
    let (s_phi, s_theta, s_wl, s_u, s_v) = (
        sp[0] as usize,
        sp[1] as usize,
        sp[2] as usize,
        sp[3] as usize,
        sp[4] as usize,
    );
    if s_wl != n_wavelengths {
        return None;
    }

    // Average spectra across all directions and VNDF samples
    // Index: [phi][theta][wl][u][v]
    let mut avg_rgb = [0.0f64; 3];
    let mut count = 0u64;
    let uv_center = s_u / 2; // sample near center of VNDF
    let uv_range = 1.max(s_u / 8); // average a small region around center
    for phi in 0..s_phi {
        for theta in 0..s_theta {
            for du in 0..uv_range {
                for dv in 0..uv_range {
                    let u = (uv_center - uv_range / 2 + du).min(s_u - 1);
                    let v = (uv_center - uv_range / 2 + dv).min(s_v - 1);
                    for (ch, &wl_idx) in rgb_idx.iter().enumerate() {
                        let idx =
                            ((phi * s_theta + theta) * s_wl + wl_idx) * s_u * s_v + u * s_v + v;
                        if idx < spec.len() {
                            avg_rgb[ch] += spec[idx].max(0.0) as f64;
                        }
                    }
                    count += 1;
                }
            }
        }
    }
    if count > 0 {
        for ch in 0..3 {
            avg_rgb[ch] /= count as f64;
        }
    }
    let avg_rgb = [avg_rgb[0] as f32, avg_rgb[1] as f32, avg_rgb[2] as f32];

    // Estimate roughness: high avg luminance relative to peak = smooth specular
    let roughness = if normal_lum > 0.01 {
        let spread = avg_lum / normal_lum;
        spread.sqrt().clamp(0.01, 1.0)
    } else {
        0.5
    };

    // Clamp albedo (VNDF-weighted values can exceed 1)
    let avg_rgb = [
        avg_rgb[0].clamp(0.0, 1.0),
        avg_rgb[1].clamp(0.0, 1.0),
        avg_rgb[2].clamp(0.0, 1.0),
    ];

    // Determine metallicness:
    // - Metals: high normal_lum with strong specular peak (low roughness)
    // - Diffuse: low normal_lum relative to avg, spread out
    let is_metallic = normal_lum > avg_lum * 5.0 && roughness < 0.5;

    if is_metallic {
        let approx_ior = |f0: f32| -> (f32, f32) {
            let f0c = f0.clamp(0.001, 0.999);
            let k = 2.0 * (f0c / (1.0 - f0c)).sqrt();
            (1.0, k)
        };
        let (er, kr) = approx_ior(avg_rgb[0]);
        let (eg, kg) = approx_ior(avg_rgb[1]);
        let (eb, kb) = approx_ior(avg_rgb[2]);
        Some(BsdfApprox {
            albedo: avg_rgb,
            roughness,
            is_metallic: true,
            eta: [er, eg, eb],
            k: [kr, kg, kb],
        })
    } else {
        Some(BsdfApprox {
            albedo: avg_rgb,
            roughness,
            is_metallic: false,
            eta: [0.0; 3],
            k: [0.0; 3],
        })
    }
}
