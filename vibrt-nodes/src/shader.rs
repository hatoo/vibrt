//! Validated, typed shader DAG. C++ implements every numerical operation.
use anyhow::{bail, ensure, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

pub const MAX_NODES: usize = 256;
pub const OUTPUTS: [&str; 32] = [
    "base_color",
    "roughness",
    "metallic",
    "ior",
    "transmission",
    "emission",
    "emission_strength",
    "alpha",
    "specular_ior_level",
    "anisotropy",
    "tangent_rotation",
    "coat_weight",
    "coat_roughness",
    "coat_ior",
    "sheen_weight",
    "sheen_roughness",
    "sheen_tint",
    "sss_weight",
    "sss_radius",
    "translucent_weight",
    "hair_offset",
    "hair_roughness_u",
    "hair_roughness_v",
    "mix_fac",
    "normal",
    "tangent",
    "bump_height",
    "bump_strength",
    "bump_distance",
    "displacement",
    "weight",
    "reserved",
];
// Order is the ABI in native/shader.h. Inputs are explicit typed conversions.
pub const OPS: [(&str, usize); 40] = [
    ("constant", 0),
    ("context", 0),
    ("convert", 1),
    ("math", 3),
    ("vector_math", 4),
    ("mix", 3),
    ("invert", 2),
    ("hsv", 5),
    ("gamma", 2),
    ("bright_contrast", 3),
    ("combine", 3),
    ("separate", 1),
    ("clamp", 3),
    ("map_range", 6),
    ("mapping", 4),
    ("vector_rotate", 5),
    ("noise", 9),
    ("voronoi", 9),
    ("wave", 7),
    ("checker", 4),
    ("brick", 10),
    ("gradient", 1),
    ("magic", 3),
    ("white_noise", 2),
    ("gabor", 6),
    ("image", 1),
    ("curve", 2),
    ("ramp", 1),
    ("blackbody", 1),
    ("wavelength", 1),
    ("fresnel", 2),
    ("layer_weight", 2),
    ("normal_map", 2),
    ("normal", 1),
    ("vector_transform", 1),
    ("squeeze", 3),
    ("camera", 0),
    ("light_falloff", 2),
    ("radial_tiling", 3),
    ("tangent", 0),
];

#[derive(Clone, Deserialize)]
pub struct ShaderGraph {
    pub nodes: Vec<ShaderNode>,
    pub outputs: BTreeMap<String, u32>,
}

#[derive(Clone, Deserialize)]
pub struct ShaderNode {
    pub op: String,
    #[serde(default)]
    pub inputs: Vec<u32>,
    #[serde(default)]
    pub modes: [u32; 4],
    #[serde(default)]
    pub params: [f32; 4],
    #[serde(default)]
    pub texture: Option<u32>,
    #[serde(default)]
    pub table: Vec<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ShaderInstruction {
    pub op: u32,
    pub inputs: [u32; 12],
    pub modes: [u32; 4],
    pub params: [f32; 4],
    pub width: i32,
    pub height: i32,
    pub table_len: u32,
    pub data: u64,
}
const _: () = assert!(std::mem::size_of::<ShaderInstruction>() == 104);

impl ShaderGraph {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.nodes.is_empty() && self.nodes.len() <= MAX_NODES,
            "Shader graph needs 1..={MAX_NODES} nodes"
        );
        ensure!(!self.outputs.is_empty(), "Shader graph has no outputs");
        for (name, &id) in &self.outputs {
            ensure!(
                OUTPUTS.contains(&name.as_str()) && (id as usize) < self.nodes.len(),
                "Invalid shader output {name} -> {id}"
            );
        }
        for (i, node) in self.nodes.iter().enumerate() {
            let (_, arity) = OPS
                .iter()
                .find(|(name, _)| *name == node.op)
                .ok_or_else(|| anyhow::anyhow!("Unknown shader operation {}", node.op))?;
            ensure!(
                node.inputs.len() == *arity,
                "Shader node {i} {} expects {arity} inputs",
                node.op
            );
            ensure!(
                node.inputs.iter().all(|&r| (r as usize) < i),
                "Shader node {i} has a forward/cyclic reference"
            );
            ensure!(
                node.params
                    .iter()
                    .chain(node.table.iter())
                    .all(|x| x.is_finite()),
                "Non-finite shader parameter at node {i}"
            );
            ensure!(
                node.table.len() <= 1_048_576,
                "Shader lookup table too large"
            );
            let bounds = match node.op.as_str() {
                "context" => [32, 1, 1, 1],
                "convert" => [4, 1, 1, 1],
                "math" => [41, 2, 1, 1],
                "vector_math" => [30, 2, 1, 1],
                "mix" => [19, 3, 2, 2],
                "combine" | "separate" => [3, 4, 1, 1],
                "map_range" => [4, 2, 2, 1],
                "mapping" => [4, 1, 1, 1],
                "vector_rotate" => [5, 2, 1, 1],
                "noise" => [5, 5, 2, 2],
                "voronoi" => [5, 5, 4, 10],
                "wave" => [2, 4, 4, 3],
                "gradient" => [7, 1, 1, 1],
                "magic" => [11, 2, 1, 1],
                "white_noise" => [5, 2, 1, 1],
                "gabor" => [2, 3, 1, 1],
                "image" => [4, 4, 4, 2],
                "curve" => [3, 2, 1, 1],
                "ramp" => [2, 2, 1, 1],
                "clamp" | "layer_weight" => [2, 1, 1, 1],
                "normal_map" => [5, 2, 1, 1],
                "normal" => [2, 1, 1, 1],
                "vector_transform" => [3, 3, 3, 1],
                "camera" | "light_falloff" => [3, 1, 1, 1],
                "radial_tiling" => [4, 2, 1, 1],
                "tangent" => [2, 3, 1, 1],
                _ => [u32::MAX; 4],
            };
            ensure!(
                node.modes.iter().zip(bounds).all(|(&v, limit)| v < limit),
                "Invalid mode for shader node {i} {}",
                node.op
            );
            if matches!(node.op.as_str(), "noise" | "voronoi" | "white_noise") {
                ensure!(node.modes[0] >= 1, "Texture dimension must be 1..4");
            }
            if matches!(node.op.as_str(), "curve" | "ramp") {
                ensure!(
                    node.table.len() >= 8 && node.table.len() % 4 == 0,
                    "Shader LUT must have at least two RGBA samples"
                );
            }
        }
        Ok(())
    }

    pub fn pack(
        &self,
        textures: &[(u64, i32, i32)],
        tables: &[u64],
    ) -> Result<Vec<ShaderInstruction>> {
        self.validate()?;
        ensure!(
            tables.len() == self.nodes.len(),
            "Missing shader lookup-table bindings"
        );
        let mut packed = Vec::with_capacity(self.nodes.len() + self.outputs.len());
        for (i, n) in self.nodes.iter().enumerate() {
            let mut r = ShaderInstruction {
                op: OPS.iter().position(|(name, _)| *name == n.op).unwrap() as u32,
                modes: n.modes,
                params: n.params,
                table_len: n.table.len() as u32,
                data: tables[i],
                ..Default::default()
            };
            r.inputs[..n.inputs.len()].copy_from_slice(&n.inputs);
            if n.op == "image" {
                let Some(&(data, w, h)) = n.texture.and_then(|id| textures.get(id as usize)) else {
                    bail!("Missing image binding at shader node {i}")
                };
                ensure!(
                    data != 0 && w > 0 && h > 0,
                    "Invalid shader texture dimensions"
                );
                r.data = data;
                r.width = w;
                r.height = h;
            } else if !n.table.is_empty() {
                ensure!(r.data != 0, "Null shader lookup-table binding");
            }
            packed.push(r);
        }
        for (name, &id) in &self.outputs {
            let mut r = ShaderInstruction {
                op: 255,
                ..Default::default()
            };
            r.inputs[0] = id;
            r.modes[0] = OUTPUTS.iter().position(|n| *n == name).unwrap() as u32;
            packed.push(r);
        }
        Ok(packed)
    }
}

extern "C" {
    fn vibrt_shader_eval(
        nodes: *const ShaderInstruction,
        count: u32,
        context: *const f32,
        output: *mut f32,
    );
}

/// Evaluate with owned, validated bindings. No pointers from Python are accepted.
pub fn evaluate(graph: &ShaderGraph, contexts: &[[f32; 128]]) -> Result<Vec<[[f32; 4]; 32]>> {
    ensure!(
        contexts.len() <= 16_777_216 && contexts.iter().flatten().all(|v| v.is_finite()),
        "Invalid shader contexts"
    );
    let tables: Vec<_> = graph
        .nodes
        .iter()
        .map(|n| n.table.as_ptr() as u64)
        .collect();
    let instructions = graph.pack(&[], &tables)?;
    let mut result = Vec::with_capacity(contexts.len());
    for context in contexts {
        let mut output = [[0.0; 4]; 32];
        // SAFETY: graph validation guarantees bounded slots/outputs and live LUTs.
        unsafe {
            vibrt_shader_eval(
                instructions.as_ptr(),
                instructions.len() as u32,
                context.as_ptr(),
                output.as_mut_ptr().cast(),
            );
        }
        result.push(output);
    }
    Ok(result)
}
