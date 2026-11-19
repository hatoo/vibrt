//! Validated, typed shader DAG. C++ implements every numerical operation.
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MAX_NODES: usize = 256;
// Packed op: opcode in bits 0..7, evaluation groups in 8..15, result slot in 16..23.
// Keep these groups in sync with native/shader.h.
const EVAL_ALL: u32 = 1;
const EVAL_BUMP: u32 = 2;
const EVAL_ALPHA: u32 = 4;
const EVAL_EMISSION: u32 = 8;
const EVAL_MIX: u32 = 16;

fn output_group(name: &str) -> u32 {
    EVAL_ALL
        | match name {
            "bump_height" => EVAL_BUMP,
            "alpha" => EVAL_ALPHA,
            "emission" | "emission_strength" => EVAL_EMISSION,
            "mix_fac" => EVAL_MIX,
            _ => 0,
        }
}
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

#[derive(Clone, Deserialize, Serialize)]
pub struct ShaderGraph {
    pub nodes: Vec<ShaderNode>,
    pub outputs: BTreeMap<String, u32>,
}

#[derive(Clone, Deserialize, Serialize)]
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

    /// Assign reusable slots and mark the dependencies of each evaluation group.
    /// Outputs are written immediately after their producer, so they need no slot lifetime.
    fn schedule(&self) -> (Vec<u32>, Vec<u32>, usize) {
        let mut groups = vec![0; self.nodes.len()];
        for (name, &id) in &self.outputs {
            groups[id as usize] |= output_group(name);
        }
        for i in (0..self.nodes.len()).rev() {
            for &input in &self.nodes[i].inputs {
                groups[input as usize] |= groups[i];
            }
        }
        let mut uses = vec![0; self.nodes.len()];
        for (node, &group) in self.nodes.iter().zip(&groups) {
            if group != 0 {
                for &input in &node.inputs {
                    uses[input as usize] += 1;
                }
            }
        }
        let mut slots = vec![0; self.nodes.len()];
        let mut free = Vec::new();
        let mut capacity = 0;
        for (i, node) in self.nodes.iter().enumerate() {
            if groups[i] == 0 {
                continue;
            }
            // The evaluator reads all operands before writing the result.
            for &input in &node.inputs {
                let input = input as usize;
                uses[input] -= 1;
                if uses[input] == 0 {
                    free.push(slots[input]);
                }
            }
            slots[i] = free.pop().unwrap_or_else(|| {
                let slot = capacity;
                capacity += 1;
                slot as u32
            });
            if uses[i] == 0 {
                free.push(slots[i]);
            }
        }
        (groups, slots, capacity)
    }

    pub fn required_stack_size(&self) -> Result<usize> {
        self.validate()?;
        Ok(self.schedule().2)
    }

    /// Context slots read explicitly or internally by native/shader.h operations.
    pub fn required_contexts(&self) -> Result<u32> {
        self.validate()?;
        let (groups, _, _) = self.schedule();
        let mut mask = 0;
        for (node, group) in self.nodes.iter().zip(groups) {
            if group == 0 {
                continue;
            }
            mask |= match node.op.as_str() {
                "context" => 1 << node.modes[0],
                "image" if node.modes[2] == 1 => 1 << 13,
                "fresnel" | "layer_weight" => (1 << 6) | (1 << 16),
                "normal_map" => (1 << 4) | (1 << 7) | (1 << 16) | (1 << 17) | (0x3f << 22),
                "vector_transform" => 0x1ff << 22,
                "camera" => 1 << 12,
                "light_falloff" => 1 << 18,
                "tangent" => (1 << 1) | (1 << 4) | (1 << 7) | (0x3f << 22),
                _ => 0,
            };
        }
        Ok(mask)
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
        let (groups, slots, _) = self.schedule();
        let mut packed = Vec::with_capacity(self.nodes.len() + self.outputs.len());
        for (i, n) in self.nodes.iter().enumerate() {
            if groups[i] == 0 {
                continue;
            }
            let mut r = ShaderInstruction {
                op: OPS.iter().position(|(name, _)| *name == n.op).unwrap() as u32
                    | (groups[i] << 8)
                    | (slots[i] << 16),
                modes: n.modes,
                params: n.params,
                table_len: n.table.len() as u32,
                data: tables[i],
                ..Default::default()
            };
            for (operand, &input) in r.inputs.iter_mut().zip(&n.inputs) {
                *operand = slots[input as usize];
            }
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
            for (name, _) in self.outputs.iter().filter(|(_, id)| **id as usize == i) {
                let mut output = ShaderInstruction {
                    op: 255 | (output_group(name) << 8),
                    ..Default::default()
                };
                output.inputs[0] = slots[i];
                output.modes[0] = OUTPUTS.iter().position(|n| *n == name).unwrap() as u32;
                packed.push(output);
            }
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
        requested: u32,
    );
}

/// Evaluate with owned, validated bindings. No pointers from Python are accepted.
pub fn evaluate(graph: &ShaderGraph, contexts: &[[f32; 128]]) -> Result<Vec<[[f32; 4]; 32]>> {
    evaluate_selected(graph, contexts, EVAL_ALL)
}

fn evaluate_selected(
    graph: &ShaderGraph,
    contexts: &[[f32; 128]],
    requested: u32,
) -> Result<Vec<[[f32; 4]; 32]>> {
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
                requested,
            );
        }
        result.push(output);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn image_address_modes_preserve_negative_and_repeated_coordinates() {
        for (width, height) in [(1, 1), (2, 4), (3, 5), (4, 2)] {
            let pixels: Vec<f32> = (0..width * height)
                .flat_map(|i| [i as f32 + 1.0, 0.25, 0.5, 1.0])
                .collect();
            for extension in 0..4 {
                let graph: ShaderGraph = serde_json::from_value(json!({"nodes":[
                    {"op":"context","modes":[0,0,0,0]},
                    {"op":"image","inputs":[0],"texture":0,"modes":[1,extension,0,0]}
                ],"outputs":{"base_color":1}}))
                .unwrap();
                let packed = graph
                    .pack(&[(pixels.as_ptr() as u64, width, height)], &[0, 0])
                    .unwrap();
                let wrap = |i: i32, n: i32| match extension {
                    0 => i.rem_euclid(n),
                    3 => {
                        let j = i.rem_euclid(2 * n);
                        if j < n {
                            j
                        } else {
                            2 * n - 1 - j
                        }
                    }
                    _ => i.clamp(0, n - 1),
                };
                for x in [
                    -2 * width - 1,
                    -width,
                    -1,
                    0,
                    width - 1,
                    width,
                    width + 1,
                    2 * width + 1,
                ] {
                    for y in [
                        -2 * height - 1,
                        -height,
                        -1,
                        0,
                        height - 1,
                        height,
                        height + 1,
                        2 * height + 1,
                    ] {
                        let mut context = [0.0; 128];
                        context[0] = (x as f32 + 0.25) / width as f32;
                        context[1] = (y as f32 + 0.25) / height as f32;
                        let mut output = [[0.0; 4]; 32];
                        // SAFETY: pack validates indices; the CPU image and all
                        // fixed-size input/output arrays stay alive during FFI.
                        unsafe {
                            vibrt_shader_eval(
                                packed.as_ptr(),
                                packed.len() as u32,
                                context.as_ptr(),
                                output.as_mut_ptr().cast(),
                                EVAL_ALL,
                            );
                        }
                        let expected =
                            if extension == 2 && (x < 0 || x >= width || y < 0 || y >= height) {
                                0.0
                            } else {
                                (wrap(y, height) * width + wrap(x, width)) as f32 + 1.0
                            };
                        assert_eq!(
                            output[0][0], expected,
                            "{width}x{height}, extension={extension}, ({x},{y})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn selected_outputs_preserve_shared_dependencies_and_reused_slots() {
        let graph: ShaderGraph = serde_json::from_value(json!({"nodes":[
            {"op":"context","modes":[0,0,0,0]},
            {"op":"context","modes":[1,0,0,0]},
            {"op":"math","inputs":[0,1,0],"modes":[0,0,0,0]},
            {"op":"math","inputs":[2,1,0],"modes":[2,0,0,0]},
            {"op":"math","inputs":[2,0,0],"modes":[1,0,0,0]},
            {"op":"image","inputs":[0],"texture":99}
        ],"outputs":{"alpha":0,"base_color":3,"emission":3,"bump_height":4,"mix_fac":2}}))
        .unwrap();
        let mut context = [0.0; 128];
        context[0] = 2.0;
        context[4] = 3.0;
        let all = evaluate(&graph, &[context]).unwrap()[0];
        assert_eq!(all[0][0], 15.0);
        assert_eq!(all[5][0], 15.0);
        assert_eq!(all[7][0], 2.0);
        assert_eq!(all[23][0], 5.0);
        assert_eq!(all[26][0], 3.0);
        assert_eq!(graph.required_stack_size().unwrap(), 3);
        for (group, sockets) in [
            (EVAL_BUMP, vec![26]),
            (EVAL_ALPHA, vec![7]),
            (EVAL_EMISSION | EVAL_ALPHA, vec![5, 7]),
            (EVAL_MIX, vec![23]),
        ] {
            let selected = evaluate_selected(&graph, &[context], group).unwrap()[0];
            for i in 0..32 {
                assert_eq!(
                    selected[i],
                    if sockets.contains(&i) {
                        all[i]
                    } else {
                        [0.0; 4]
                    }
                );
            }
        }
    }

    #[test]
    fn long_chain_needs_only_two_live_slots() {
        let mut nodes = vec![json!({"op":"context","modes":[0,0,0,0]})];
        for i in 1..MAX_NODES {
            nodes.push(json!({"op":"math","inputs":[i-1,0,0],"modes":[0,0,0,0]}));
        }
        let graph: ShaderGraph = serde_json::from_value(json!({
            "nodes":nodes,"outputs":{"base_color":MAX_NODES-1,"bump_height":MAX_NODES-1}
        }))
        .unwrap();
        let mut context = [0.0; 128];
        context[0] = 1.0;
        assert_eq!(graph.required_stack_size().unwrap(), 2);
        assert_eq!(evaluate(&graph, &[context]).unwrap()[0][0][0], 256.0);
        assert_eq!(
            evaluate_selected(&graph, &[context], EVAL_BUMP).unwrap()[0][26][0],
            256.0
        );
    }

    #[test]
    fn implicit_context_reads_match_the_prepared_mask() {
        let mut cases = Vec::new();
        for (op, modes) in [
            ("fresnel", 1),
            ("layer_weight", 2),
            ("normal_map", 5),
            ("camera", 3),
            ("light_falloff", 3),
            ("tangent", 2),
        ] {
            for mode in 0..modes {
                cases.push((op, [mode, 0, 0, 0]));
            }
        }
        for kind in 0..3 {
            for from in 0..3 {
                for to in 0..3 {
                    cases.push(("vector_transform", [kind, from, to, 0]));
                }
            }
        }
        let context = std::array::from_fn(|i| 0.25 + (i as f32 * 0.7).sin());
        for (op, modes) in cases {
            let arity = OPS.iter().find(|(name, _)| *name == op).unwrap().1;
            let graph: ShaderGraph = serde_json::from_value(json!({"nodes":[
                {"op":"constant","params":[0.2,0.3,0.8,0.5]},
                {"op":op,"inputs":vec![0;arity],"modes":modes}
            ],"outputs":{"base_color":1}}))
            .unwrap();
            let mask = graph.required_contexts().unwrap();
            let mut prepared = [-987.0; 128];
            for slot in 0..32 {
                if mask & (1 << slot) != 0 {
                    prepared[slot * 4..slot * 4 + 4]
                        .copy_from_slice(&context[slot * 4..slot * 4 + 4]);
                }
            }
            assert_eq!(
                evaluate(&graph, &[context]).unwrap(),
                evaluate(&graph, &[prepared]).unwrap(),
                "{op} {modes:?} reads a context slot missing from {mask:#x}"
            );
        }
    }
}
