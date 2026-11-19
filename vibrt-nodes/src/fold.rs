//! Constant propagation controls the graph; C++ computes all node values.
use crate::color::{apply, ColorOp};
use crate::graph::{ColorFactor, ColorGraph, ColorNode};

pub fn constants(graph: &ColorGraph) -> Vec<Option<[f32; 3]>> {
    let mut values = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        values.push(evaluate(node, &values));
    }
    values
}

fn evaluate(node: &ColorNode, prev: &[Option<[f32; 3]>]) -> Option<[f32; 3]> {
    let input = |index: u32| prev.get(index as usize).copied().flatten();
    let unary = |op, value: [f32; 3]| -> Option<[f32; 3]> {
        apply(&op, &value, None, None).ok()?.try_into().ok()
    };
    match node {
        ColorNode::Const { rgb } => Some(*rgb),
        ColorNode::Mix {
            a,
            b,
            fac,
            blend,
            clamp,
        } => {
            let a = input(*a)?;
            let b = input(*b)?;
            let fac = match fac {
                ColorFactor::Const(value) => *value,
                ColorFactor::Node { node } => unary(ColorOp::Luminance {}, input(*node)?)?[0],
            };
            apply(
                &ColorOp::Mix {
                    fac,
                    blend: blend.clone(),
                    clamp: *clamp,
                },
                &a,
                Some(&b),
                None,
            )
            .ok()?
            .try_into()
            .ok()
        }
        ColorNode::Math {
            input: i,
            op,
            b,
            c,
            clamp,
            swap,
        } => unary(
            ColorOp::Math {
                op: op.clone(),
                b: *b,
                c: *c,
                clamp: *clamp,
                swap: *swap,
            },
            input(*i)?,
        ),
        ColorNode::Invert { input: i, fac } => unary(ColorOp::Invert { fac: *fac }, input(*i)?),
        ColorNode::HueSat {
            input: i,
            hue,
            saturation,
            value,
            fac,
        } => unary(
            ColorOp::HueSat {
                hue: *hue,
                saturation: *saturation,
                value: *value,
                fac: *fac,
                clamp_negative: false,
            },
            input(*i)?,
        ),
        ColorNode::BrightContrast {
            input: i,
            bright,
            contrast,
        } => unary(
            ColorOp::BrightContrast {
                bright: *bright,
                contrast: *contrast,
            },
            input(*i)?,
        ),
        ColorNode::RgbCurve { input: i, lut } if lut.len() == 768 => unary(
            ColorOp::RgbCurve {
                lut: lut.clone(),
                low: 0.0,
                high: 1.0,
                fac: 1.0,
                extrapolate: false,
            },
            input(*i)?,
        ),
        // Vertex color, object random and images depend on a shading point.
        // Preserve unsupported/malformed nodes for the upload validator.
        _ => None,
    }
}
