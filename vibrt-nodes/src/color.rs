//! Checked, batched C++ color evaluation. No color equations live in Rust.

use crate::graph::{parse_blend, parse_math_op};
use crate::types::*;
use anyhow::{ensure, Result};
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ColorOp {
    ScalarMath {
        op: String,
        #[serde(default)]
        b: f32,
        #[serde(default)]
        c: f32,
        #[serde(default)]
        clamp: bool,
    },
    Mix {
        #[serde(default = "half")]
        fac: f32,
        #[serde(default = "mix")]
        blend: String,
        #[serde(default)]
        clamp: bool,
    },
    Math {
        op: String,
        #[serde(default)]
        b: f32,
        #[serde(default)]
        c: f32,
        #[serde(default)]
        swap: bool,
        #[serde(default)]
        clamp: bool,
    },
    Invert {
        #[serde(default = "one")]
        fac: f32,
    },
    HueSat {
        #[serde(default = "half")]
        hue: f32,
        #[serde(default = "one")]
        saturation: f32,
        #[serde(default = "one")]
        value: f32,
        #[serde(default = "one")]
        fac: f32,
        #[serde(default)]
        clamp_negative: bool,
    },
    RgbCurve {
        lut: Vec<f32>,
        #[serde(default)]
        low: f32,
        #[serde(default = "one")]
        high: f32,
        #[serde(default = "one")]
        fac: f32,
        #[serde(default)]
        extrapolate: bool,
    },
    BrightContrast {
        #[serde(default)]
        bright: f32,
        #[serde(default)]
        contrast: f32,
    },
    ColorRamp {
        lut: Vec<f32>,
        #[serde(default)]
        luminance: bool,
    },
    Gamma {
        gamma: f32,
        #[serde(default = "yes")]
        preserve_negative: bool,
    },
    Clamp {
        min: f32,
        max: f32,
    },
    RgbToHsv {},
    HsvToRgb {},
    Luminance {},
}
fn one() -> f32 {
    1.0
}
fn half() -> f32 {
    0.5
}
fn yes() -> bool {
    true
}
fn mix() -> String {
    "mix".into()
}

#[repr(C)]
struct NativeColor {
    tag: u32,
    mode: u32,
    params: [f32; 8],
    lut: *const f32,
    lut_len: u64,
}
const _: () = assert!(std::mem::size_of::<NativeColor>() == 56);
extern "C" {
    fn vibrt_color_batch(
        op: *const NativeColor,
        a: *const f32,
        a_stride: usize,
        b: *const f32,
        b_stride: usize,
        factors: *const f32,
        factor_stride: usize,
        count: usize,
        out: *mut f32,
    );
}

struct CompiledColor<'a> {
    native: NativeColor,
    _owner: &'a ColorOp,
}
// C++ reads the immutable LUT owned by _owner. All workers join before its
// borrow ends; output slices are disjoint, and no Python/renderer state is used.
unsafe impl Sync for CompiledColor<'_> {}
impl CompiledColor<'_> {
    fn native(&self) -> &NativeColor {
        &self.native
    }
}

impl ColorOp {
    fn compile(&self) -> Result<CompiledColor<'_>> {
        let mut native = NativeColor {
            tag: 0,
            mode: 0,
            params: [0.0; 8],
            lut: std::ptr::null(),
            lut_len: 0,
        };
        let p = &mut native.params;
        match self {
            Self::ScalarMath { op, b, c, clamp } => {
                native.tag = 16;
                native.mode = parse_scalar_op(op)?;
                p[..3].copy_from_slice(&[*b, *c, u8::from(*clamp) as f32]);
            }
            Self::Mix { fac, blend, clamp } => {
                native.tag = COLOR_NODE_MIX;
                native.mode = parse_blend(blend)?;
                p[..2].copy_from_slice(&[*fac, u8::from(*clamp) as f32]);
            }
            Self::Math {
                op,
                b,
                c,
                swap,
                clamp,
            } => {
                native.tag = COLOR_NODE_MATH;
                native.mode = parse_math_op(op)?;
                p[..4].copy_from_slice(&[*b, *c, u8::from(*swap) as f32, u8::from(*clamp) as f32]);
            }
            Self::Invert { fac } => {
                native.tag = COLOR_NODE_INVERT;
                p[0] = *fac;
            }
            Self::HueSat {
                hue,
                saturation,
                value,
                fac,
                clamp_negative,
            } => {
                native.tag = COLOR_NODE_HUE_SAT;
                p[..5].copy_from_slice(&[
                    *hue,
                    *saturation,
                    *value,
                    *fac,
                    u8::from(*clamp_negative) as f32,
                ]);
            }
            Self::RgbCurve {
                lut,
                low,
                high,
                fac,
                extrapolate,
            } => {
                ensure!(
                    lut.len() >= 6 && lut.len().is_multiple_of(3),
                    "RGB curve needs equal channel tables with at least two samples"
                );
                ensure!(high > low, "RGB curve domain must have positive width");
                native.tag = COLOR_NODE_RGB_CURVE;
                native.lut = lut.as_ptr();
                native.lut_len = lut.len() as u64;
                p[..4].copy_from_slice(&[*low, *high, *fac, u8::from(*extrapolate) as f32]);
            }
            Self::ColorRamp { lut, luminance } => {
                ensure!(
                    lut.len() >= 3 && lut.len().is_multiple_of(3),
                    "Color ramp needs RGB stops"
                );
                native.tag = COLOR_NODE_COLOR_RAMP;
                native.lut = lut.as_ptr();
                native.lut_len = lut.len() as u64;
                p[0] = u8::from(*luminance) as f32;
            }
            Self::BrightContrast { bright, contrast } => {
                native.tag = COLOR_NODE_BRIGHT_CONTRAST;
                p[..2].copy_from_slice(&[*bright, *contrast]);
            }
            Self::Gamma {
                gamma,
                preserve_negative,
            } => {
                native.tag = 11;
                p[..2].copy_from_slice(&[*gamma, u8::from(*preserve_negative) as f32]);
            }
            Self::Clamp { min, max } => {
                native.tag = 12;
                p[..2].copy_from_slice(&[*min, *max]);
            }
            Self::RgbToHsv {} => native.tag = 13,
            Self::HsvToRgb {} => native.tag = 14,
            Self::Luminance {} => native.tag = 15,
        }
        ensure!(
            p.iter().all(|v| v.is_finite()),
            "Color parameters must be finite"
        );
        if let Self::RgbCurve { lut, .. } | Self::ColorRamp { lut, .. } = self {
            ensure!(
                lut.len() <= 3_145_728 && lut.iter().all(|v| v.is_finite()),
                "Invalid color lookup table"
            );
        }
        Ok(CompiledColor {
            native,
            _owner: self,
        })
    }
}

/// RGB inputs and scalar factors may each contain one sample (broadcast) or
/// the full batch. No borrowed pointer survives this synchronous call.
pub fn apply(
    op: &ColorOp,
    a: &[f32],
    b: Option<&[f32]>,
    factors: Option<&[f32]>,
) -> Result<Vec<f32>> {
    ensure!(
        !a.is_empty() && a.len().is_multiple_of(3),
        "Expected packed RGB input"
    );
    if let Some(b) = b {
        ensure!(
            !b.is_empty() && b.len().is_multiple_of(3),
            "Expected packed second RGB input"
        );
    }
    ensure!(
        !matches!(op, ColorOp::Mix { .. }) || b.is_some(),
        "Mix requires a second RGB input"
    );
    ensure!(
        factors.is_none() || matches!(op, ColorOp::Mix { .. }),
        "Factor arrays require a Mix operation"
    );
    let count = (a.len() / 3)
        .max(b.map_or(1, |b| b.len() / 3))
        .max(factors.map_or(1, |f| f.len()));
    ensure!(count <= 16_777_216, "Color batch exceeds 16 megapixels");
    for len in [
        a.len() / 3,
        b.map_or(1, |b| b.len() / 3),
        factors.map_or(1, |f| f.len()),
    ] {
        ensure!(len == 1 || len == count, "Incompatible color input lengths");
    }
    ensure!(
        a.iter()
            .chain(b.unwrap_or_default())
            .chain(factors.unwrap_or_default())
            .all(|v| v.is_finite()),
        "Color inputs must be finite"
    );
    let compiled = op.compile()?;
    let mut out = vec![0.0; count * 3];
    let batch = |start: usize, dst: &mut [f32]| {
        let rgb_input = |input: &[f32]| {
            let stride = if input.len() == 3 { 0 } else { 3 };
            (input[start * stride..].as_ptr(), stride)
        };
        let (ap, ast) = rgb_input(a);
        let (bp, bst) = b.map_or((std::ptr::null(), 0), rgb_input);
        let (fp, fst) = factors.map_or((std::ptr::null(), 0), |f| {
            let stride = usize::from(f.len() != 1);
            (f[start * stride..].as_ptr(), stride)
        });
        // Shapes and LUT sizes validated above. C++ writes exactly dst.len()
        // floats, only reads borrowed inputs, and cannot retain their pointers.
        unsafe {
            vibrt_color_batch(
                compiled.native(),
                ap,
                ast,
                bp,
                bst,
                fp,
                fst,
                dst.len() / 3,
                dst.as_mut_ptr(),
            );
        }
    };
    if count < 8192 {
        batch(0, &mut out);
    } else {
        out.par_chunks_mut(4096 * 3)
            .enumerate()
            .for_each(|(i, dst)| batch(i * 4096, dst));
    }
    ensure!(
        out.iter().all(|v| v.is_finite()),
        "Color operation produced non-finite values"
    );
    Ok(out)
}

fn parse_scalar_op(op: &str) -> Result<u32> {
    Ok(match op {
        "logarithm" => 8,
        "less_than" => 9,
        "greater_than" => 10,
        "compare" => 11,
        "modulo" => 12,
        "smooth_min" => 13,
        "smooth_max" => 14,
        "arctan2" => 15,
        "sqrt" => 16,
        "inverse_sqrt" => 17,
        "absolute" => 18,
        "exponent" => 19,
        "sign" => 20,
        "round" => 21,
        "floor" => 22,
        "ceil" => 23,
        "fract" => 24,
        "trunc" => 25,
        "sine" => 26,
        "cosine" => 27,
        "tangent" => 28,
        "arcsine" => 29,
        "arccosine" => 30,
        "arctangent" => 31,
        "hyperbolic_sine" => 32,
        "hyperbolic_cosine" => 33,
        "hyperbolic_tangent" => 34,
        "radians" => 35,
        "degrees" => 36,
        other => return parse_math_op(other),
    })
}
