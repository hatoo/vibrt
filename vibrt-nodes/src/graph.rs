use crate::types::*;
use anyhow::{anyhow, ensure, Result};
use serde::Deserialize;

/// Sequential list of colour-producing nodes. Each node reads either from
/// earlier nodes (by index) or from constants/textures; the last node's
/// value is the material's base colour. Input indices must be strictly
/// less than the owning node's index so the graph stays a DAG in
/// topological order.
#[derive(Deserialize, Clone)]
pub struct ColorGraph {
    pub nodes: Vec<ColorNode>,
    /// Index of the output node. Default: last node.
    #[serde(default)]
    pub output: Option<u32>,
}

/// Factor input for a Mix node: either a constant scalar or a reference to
/// another node's value (converted to luminance at evaluation time).
#[derive(Deserialize, Clone)]
#[serde(untagged)]
pub enum ColorFactor {
    Const(f32),
    Node { node: u32 },
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ColorNode {
    /// Literal RGB.
    Const { rgb: [f32; 3] },
    /// Sample a texture (by index into `SceneFile::textures`) with its own
    /// UV transform applied on top of the mesh UV.
    ImageTex {
        tex: u32,
        #[serde(default = "identity_uv_transform")]
        uv: [f32; 6],
    },
    /// Mix two inputs with the given blend and factor. `blend` follows
    /// Blender's naming: "mix" / "multiply" / "add" / "subtract" / "screen"
    /// / "divide" / "difference" / "darken" / "lighten" / "overlay" /
    /// "soft_light" / "linear_light".
    Mix {
        a: u32,
        b: u32,
        #[serde(default = "default_mix_fac")]
        fac: ColorFactor,
        #[serde(default = "default_blend")]
        blend: String,
        #[serde(default)]
        clamp: bool,
    },
    /// Per-channel RGB invert with mix factor (`out = lerp(rgb, 1-rgb, fac)`).
    Invert {
        input: u32,
        #[serde(default = "one_f32")]
        fac: f32,
    },
    /// Scalar math on the input treated per-channel. Ops mirror
    /// ShaderNodeMath: "add" / "subtract" / "multiply" / "divide" /
    /// "power" / "multiply_add" / "minimum" / "maximum".
    ///
    /// `swap` flips operand order for non-commutative ops (subtract /
    /// divide / power): when true the op evaluates as `b OP input` instead
    /// of `input OP b`. Emitted when Cycles drives the second Math input
    /// (rather than the first) with the texture chain. Ignored for
    /// commutative ops.
    Math {
        input: u32,
        #[serde(default = "default_math_op")]
        op: String,
        #[serde(default)]
        b: f32,
        #[serde(default)]
        c: f32,
        #[serde(default)]
        clamp: bool,
        #[serde(default)]
        swap: bool,
    },
    /// Blender HueSaturation: shift hue, scale saturation/value, then blend
    /// with the original by `fac`. All scalars are constants (Blender
    /// interprets linked ones here too but very rare).
    HueSat {
        input: u32,
        #[serde(default = "half_f32")]
        hue: f32,
        #[serde(default = "one_f32")]
        saturation: f32,
        #[serde(default = "one_f32")]
        value: f32,
        #[serde(default = "one_f32")]
        fac: f32,
    },
    /// Per-channel RGB curve, pre-baked by the exporter into a `768`-entry
    /// LUT (256 samples × 3 channels). Each channel's entry already folds
    /// the per-channel curve through the combined curve, matching how
    /// Blender stacks them in ShaderNodeRGBCurve.
    RgbCurve {
        input: u32,
        /// 768 floats: R[0..256], G[0..256], B[0..256]. Each subarray is
        /// the curve sampled at `i/255` and clamped to [0, 1].
        lut: Vec<f32>,
    },
    /// Blender BrightContrast: `out = a * rgb + b` with Cycles' parameter
    /// convention `a = 1 + contrast`, `b = bright - contrast/2`.
    BrightContrast {
        input: u32,
        #[serde(default)]
        bright: f32,
        #[serde(default)]
        contrast: f32,
    },
    /// Blender ShaderNodeAttribute reading a per-vertex colour. The mesh must
    /// ship the matching colour attribute (exporter plumbs this via
    /// `_vertex_color_attr`); when absent the device falls back to white.
    VertexColor {},
    /// Per-instance uniform random in [0..1) — derived from the OptiX
    /// instance id at hit time. Mirrors `ShaderNodeObjectInfo.Random`. The
    /// scalar is broadcast to all three RGB channels so downstream Mix /
    /// ColorRamp consumers can use it both as a colour and as a Fac.
    ObjectRandom {},
    /// Look up a 1D RGB ramp at a Fac driven by another graph slot. Used
    /// when ColorRamp's Fac chain is non-foldable (typically
    /// `ObjectInfo.Random` feeding a per-instance palette). The exporter
    /// bakes the ramp to `lut.len() / 3` evenly-spaced linear-RGB stops.
    ColorRamp {
        input: u32,
        /// `n_stops × 3` floats: R[0..n], G[0..n], B[0..n] flattened by
        /// stop, so payload[i*3..i*3+3] is the i-th stop's linear RGB.
        lut: Vec<f32>,
    },
}

fn default_mix_fac() -> ColorFactor {
    ColorFactor::Const(0.5)
}

fn default_blend() -> String {
    "mix".to_string()
}

fn default_math_op() -> String {
    "multiply".to_string()
}

fn one_f32() -> f32 {
    1.0
}
fn half_f32() -> f32 {
    0.5
}
fn identity_uv_transform() -> [f32; 6] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

pub fn pack(
    graph: &ColorGraph,
    textures: &[crate::types::TextureBinding],
    lut_ptrs: &[u64],
    lut_lens: &[u32],
) -> Result<Vec<ColorGraphNode>> {
    graph.validate()?;
    ensure!(
        lut_ptrs.len() == graph.nodes.len() && lut_lens.len() == graph.nodes.len(),
        "Missing graph lookup-table bindings"
    );
    let mut out: Vec<ColorGraphNode> = Vec::with_capacity(graph.nodes.len());
    for (i, node) in graph.nodes.iter().enumerate() {
        let mut record = ColorGraphNode {
            tag: 0,
            payload: [0u32; 15],
        };
        flatten_one(node, i, textures, lut_ptrs, lut_lens, &mut record)?;
        out.push(record);
    }
    Ok(out)
}

fn flatten_one(
    node: &ColorNode,
    self_idx: usize,
    textures: &[crate::types::TextureBinding],
    lut_ptrs: &[u64],
    lut_lens: &[u32],
    out: &mut ColorGraphNode,
) -> Result<()> {
    let check_ref = |r: u32, field: &str| -> Result<()> {
        if (r as usize) >= self_idx {
            Err(anyhow!(
                "color_graph node {} {}: references node {} >= self",
                self_idx,
                field,
                r
            ))
        } else {
            Ok(())
        }
    };
    let p = &mut out.payload;
    match node {
        ColorNode::Const { rgb } => {
            out.tag = COLOR_NODE_CONST;
            p[0] = rgb[0].to_bits();
            p[1] = rgb[1].to_bits();
            p[2] = rgb[2].to_bits();
        }
        ColorNode::ImageTex { tex, uv } => {
            let (ptr, w, h, channels) = *textures
                .get(*tex as usize)
                .ok_or_else(|| anyhow!("color_graph ImageTex.tex = {} out of range", tex))?;
            out.tag = COLOR_NODE_IMAGE_TEX;
            p[0] = ptr as u32;
            p[1] = (ptr >> 32) as u32;
            p[2] = w as u32;
            p[3] = h as u32;
            ensure!((1..=4).contains(&channels), "Invalid texture channel count");
            p[4] = channels as u32;
            for k in 0..6 {
                p[5 + k] = uv[k].to_bits();
            }
        }
        ColorNode::Mix {
            a,
            b,
            fac,
            blend,
            clamp,
        } => {
            check_ref(*a, "a")?;
            check_ref(*b, "b")?;
            let blend_id = parse_blend(blend)?;
            out.tag = COLOR_NODE_MIX;
            p[0] = *a;
            p[1] = *b;
            p[2] = blend_id;
            match fac {
                ColorFactor::Const(v) => {
                    p[3] = 0; // fac_src = const
                    p[4] = 0;
                    p[6] = v.to_bits();
                }
                ColorFactor::Node { node } => {
                    check_ref(*node, "fac.node")?;
                    p[3] = 1; // fac_src = node
                    p[4] = *node;
                    p[6] = 0;
                }
            }
            p[5] = if *clamp { 1 } else { 0 };
        }
        ColorNode::Invert { input, fac } => {
            check_ref(*input, "input")?;
            out.tag = COLOR_NODE_INVERT;
            p[0] = *input;
            p[1] = fac.to_bits();
        }
        ColorNode::Math {
            input,
            op,
            b,
            c,
            clamp,
            swap,
        } => {
            check_ref(*input, "input")?;
            let op_id = parse_math_op(op)?;
            out.tag = COLOR_NODE_MATH;
            p[0] = *input;
            p[1] = op_id;
            p[2] = if *clamp { 1 } else { 0 };
            p[3] = b.to_bits();
            p[4] = c.to_bits();
            p[5] = if *swap { 1 } else { 0 };
        }
        ColorNode::HueSat {
            input,
            hue,
            saturation,
            value,
            fac,
        } => {
            check_ref(*input, "input")?;
            out.tag = COLOR_NODE_HUE_SAT;
            p[0] = *input;
            p[1] = hue.to_bits();
            p[2] = saturation.to_bits();
            p[3] = value.to_bits();
            p[4] = fac.to_bits();
        }
        ColorNode::RgbCurve { input, .. } => {
            check_ref(*input, "input")?;
            let lut_ptr = lut_ptrs[self_idx];
            out.tag = COLOR_NODE_RGB_CURVE;
            p[0] = *input;
            p[1] = lut_ptr as u32;
            p[2] = (lut_ptr >> 32) as u32;
        }
        ColorNode::BrightContrast {
            input,
            bright,
            contrast,
        } => {
            check_ref(*input, "input")?;
            out.tag = COLOR_NODE_BRIGHT_CONTRAST;
            p[0] = *input;
            p[1] = bright.to_bits();
            p[2] = contrast.to_bits();
        }
        ColorNode::VertexColor {} => {
            out.tag = COLOR_NODE_VERTEX_COLOR;
        }
        ColorNode::ObjectRandom {} => {
            out.tag = COLOR_NODE_OBJECT_RANDOM;
        }
        ColorNode::ColorRamp { input, .. } => {
            check_ref(*input, "input")?;
            let lut_ptr = lut_ptrs[self_idx];
            let lut_len = lut_lens[self_idx];
            out.tag = COLOR_NODE_COLOR_RAMP;
            p[0] = *input;
            p[1] = lut_ptr as u32;
            p[2] = (lut_ptr >> 32) as u32;
            p[3] = lut_len;
        }
    }
    Ok(())
}

pub fn parse_blend(s: &str) -> Result<u32> {
    Ok(match s {
        "mix" => 0,
        "multiply" => 1,
        "add" => 2,
        "subtract" => 3,
        "screen" => 4,
        "divide" => 5,
        "difference" => 6,
        "darken" => 7,
        "lighten" => 8,
        "overlay" => 9,
        "soft_light" => 10,
        "linear_light" => 11,
        "hue" => 12,
        "saturation" => 13,
        "color" => 14,
        "value" => 15,
        other => return Err(anyhow!("unknown Mix blend {:?}", other)),
    })
}

pub fn parse_math_op(s: &str) -> Result<u32> {
    Ok(match s {
        "add" => 0,
        "subtract" => 1,
        "multiply" => 2,
        "divide" => 3,
        "power" => 4,
        "multiply_add" => 5,
        "minimum" => 6,
        "maximum" => 7,
        other => return Err(anyhow!("unknown Math op {:?}", other)),
    })
}

impl ColorGraph {
    /// The interpreter uses a fixed register stack, on both CPU and GPU.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.nodes.is_empty() && self.nodes.len() <= 32,
            "Color graph requires 1..32 nodes"
        );
        ensure!(
            (self.output.unwrap_or(self.nodes.len() as u32 - 1) as usize) < self.nodes.len(),
            "Color graph output is out of range"
        );
        for node in &self.nodes {
            match node {
                ColorNode::RgbCurve { lut, .. } => ensure!(
                    lut.len() == 768 && lut.iter().all(|v| v.is_finite()),
                    "RGB curve requires 768 finite entries"
                ),
                ColorNode::ColorRamp { lut, .. } => ensure!(
                    !lut.is_empty()
                        && lut.len().is_multiple_of(3)
                        && lut.iter().all(|v| v.is_finite()),
                    "Color ramp requires finite RGB stops"
                ),
                _ => {}
            }
        }
        Ok(())
    }
}
