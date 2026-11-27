//! Owned, validated Cycles IES profiles. No borrowed pointer survives an FFI call.
use anyhow::{ensure, Result};

unsafe extern "C" {
    fn vibrt_ies_parse(text: *const u8, length: usize, output: *mut f32, capacity: usize) -> usize;
    fn vibrt_ies_pack(
        v: *const f32,
        nv: usize,
        h: *const f32,
        nh: usize,
        values: *const f32,
        output: *mut f32,
        capacity: usize,
    ) -> usize;
}

#[derive(serde::Serialize)]
pub struct Profile {
    pub cycles_processed: bool,
    pub thetas_deg: Vec<f32>,
    pub phis_deg: Vec<f32>,
    pub candelas: Vec<f32>,
    pub peak_candela: f32,
    pub peak_absolute_candela: f32,
    pub integral_norm: f32,
}

const MAX_VALUES: usize = 1_000_000;
fn collect(mut call: impl FnMut(*mut f32, usize) -> usize) -> Result<Vec<f32>> {
    let size = call(std::ptr::null_mut(), 0);
    ensure!((6..=8 * MAX_VALUES).contains(&size), "Invalid IES profile");
    let mut result = vec![0.0; size];
    ensure!(
        call(result.as_mut_ptr(), size) == size,
        "IES packing failed"
    );
    Ok(result)
}
fn validate(v: &[f32], h: &[f32], values: &[f32]) -> Result<()> {
    ensure!(
        v.len() >= 2
            && !h.is_empty()
            && v.len().saturating_mul(h.len()) == values.len()
            && values.len() <= MAX_VALUES,
        "Invalid IES dimensions"
    );
    for angles in [v, h] {
        ensure!(
            angles.iter().all(|x| x.is_finite()) && angles.windows(2).all(|p| p[0] < p[1]),
            "IES angles must be finite and increasing"
        );
    }
    ensure!(
        values.iter().all(|x| x.is_finite() && *x >= 0.0),
        "Invalid IES intensity"
    );
    Ok(())
}

/// Normalize legacy Type C tables with Cycles too; the first float is the slot offset.
pub fn pack(v: &[f32], h: &[f32], values: &[f32], peak: f32, processed: bool) -> Result<Vec<f32>> {
    validate(v, h, values)?;
    let mut packed = if processed {
        let mut data = vec![
            f32::from_bits(h.len() as u32),
            f32::from_bits(v.len() as u32),
        ];
        data.extend(h.iter().map(|x| x.to_radians()));
        data.extend(v.iter().map(|x| x.to_radians()));
        data.extend_from_slice(values);
        data
    } else {
        collect(|output, capacity| unsafe {
            vibrt_ies_pack(
                v.as_ptr(),
                v.len(),
                h.as_ptr(),
                h.len(),
                values.as_ptr(),
                output,
                capacity,
            )
        })?
    };
    let nv = packed[1].to_bits() as usize;
    let nh = packed[0].to_bits() as usize;
    ensure!(
        nv >= 2 && nh >= 2 && packed.len() == 2 + nv + nh + nv * nh,
        "Invalid processed IES table"
    );
    ensure!(peak.is_finite() && peak >= 0.0, "Invalid IES peak");
    for x in &mut packed[2 + nv + nh..] {
        *x /= peak.max(1e-20);
    }
    packed.insert(0, f32::from_bits(1));
    Ok(packed)
}

pub fn parse(text: &str) -> Result<Profile> {
    ensure!(
        text.len() <= 16 * 1024 * 1024 && !text.contains('\0'),
        "Invalid IES input size"
    );
    let text = format!(
        "\n{}",
        text.lines().map(str::trim).collect::<Vec<_>>().join("\n")
    );
    let start = text
        .find("\nTILT=")
        .ok_or_else(|| anyhow::anyhow!("Missing IES TILT header"))?;
    let (tilt, body) = text[start + 1..]
        .split_once('\n')
        .ok_or_else(|| anyhow::anyhow!("Missing IES data"))?;
    let body = body.replace(',', " ");
    let nums: Vec<f64> = body
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    ensure!(nums.iter().all(|x| x.is_finite()), "Non-finite IES data");
    let count = |x: f64| -> Result<usize> {
        ensure!(
            x >= 1.0 && x <= MAX_VALUES as f64 && x.fract() == 0.0,
            "Invalid IES count"
        );
        Ok(x as usize)
    };
    let skip = if tilt == "TILT=INCLUDE" {
        ensure!(nums.len() >= 2, "Truncated IES tilt");
        2 + 2 * count(nums[1])?
    } else {
        0
    };
    ensure!(nums.len() >= skip + 13, "Truncated IES header");
    let n = &nums[skip..];
    let nv = count(n[3])?;
    let nh = count(n[4])?;
    ensure!(
        nv >= 2 && nv.saturating_mul(nh) <= MAX_VALUES && n.len() >= 13 + nv + nh + nv * nh,
        "Invalid IES dimensions or truncated data"
    );
    let v: Vec<f32> = n[13..13 + nv].iter().map(|x| *x as f32).collect();
    let h: Vec<f32> = n[13 + nv..13 + nv + nh].iter().map(|x| *x as f32).collect();
    let values: Vec<f32> = n[13 + nv + nh..13 + nv + nh + nv * nh]
        .iter()
        .map(|x| *x as f32)
        .collect();
    validate(&v, &h, &values)?;
    let packed = collect(|output, capacity| unsafe {
        vibrt_ies_parse(text.as_ptr(), text.len(), output, capacity)
    })?;
    let nh = packed[0].to_bits() as usize;
    let nv = packed[1].to_bits() as usize;
    ensure!(
        nv >= 2 && nh >= 2 && packed.len() == 2 + nv + nh + nv * nh,
        "Invalid processed IES dimensions"
    );
    let phis_deg: Vec<f32> = packed[2..2 + nh].iter().map(|x| x.to_degrees()).collect();
    let thetas_deg: Vec<f32> = packed[2 + nh..2 + nh + nv]
        .iter()
        .map(|x| x.to_degrees())
        .collect();
    let candelas = packed[2 + nh + nv..].to_vec();
    validate(&thetas_deg, &phis_deg, &candelas)?;
    let peak = candelas.iter().copied().fold(0.0_f32, f32::max);
    Ok(Profile {
        cycles_processed: true,
        thetas_deg,
        phis_deg,
        candelas,
        peak_candela: peak,
        peak_absolute_candela: peak,
        integral_norm: 0.0,
    })
}
