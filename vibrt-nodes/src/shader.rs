//! Validated, typed shader DAG. C++ implements every numerical operation.
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

mod fold;

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
pub const OUTPUTS: [&str; 64] = [
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
    "conductor_ior",
    "toon_size",
    "toon_smooth",
    "portal_position",
    "portal_direction",
    "specular_tint",
    "coat_tint",
    "coat_normal",
    "thin_film_thickness",
    "thin_film_ior",
    "sss_scale",
    "sss_ior",
    "sss_anisotropy",
    "hair_sigma",
    "hair_random",
    "hair_coat",
    "hair_ior",
    "conductor_k",
    "diffuse_roughness",
    "hair_transmission",
    "hair_secondary",
    "hair_random_roughness",
    "hair_random_color",
    "hair_melanin",
    "hair_redness",
    "hair_tint",
    "hair_parametrization",
    "displacement_vector",
    "density",
    "absorption",
    "scattering",
    "phase",
    "volume_emission",
];
// Order is the ABI in native/shader.h. Inputs are explicit typed conversions.
pub const OPS: [(&str, usize); 52] = [
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
    ("wireframe", 1),
    ("ambient_occlusion", 3),
    ("bevel", 2),
    ("raycast", 3),
    ("attribute", 0),
    ("hair_tangent", 0),
    ("shifted_context", 0),
    ("bump", 6),
    ("normal_map_named", 3),
    ("displacement", 4),
    ("vector_displacement", 4),
    ("vector_bump", 4),
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
    #[serde(default)]
    pub attribute: String,
}

pub fn attribute_key(name: &str) -> u64 {
    name.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
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

// Whole-program forms shared with native/shader.h. One instruction writes one
// socket directly: inputs = [context slot, output socket, optional conversion].
// Retain the image's modes/params so filtering and alpha association are exact.
fn specialize_single_output(program: &[ShaderInstruction]) -> Option<ShaderInstruction> {
    let first = *program.first()?;
    let output = *program.last()?;
    if output.op & 255 != 255 {
        return None;
    }
    let (mut instruction, opcode, conversion) = match program {
        [_, _] if first.op & 255 == 0 => (first, 252, 0),
        [_, _] if first.op & 255 == 1 && first.modes[0] == 0 => (first, 253, 0),
        [_, image, ..]
            if first.op & 255 == 1
                && first.modes[0] == 0
                && image.op & 255 == 25
                && image.modes[2] == 0
                && image.params[1] == 0.0 =>
        {
            let conversion = match program {
                [_, _, _] => 0,
                [_, _, convert, _] if convert.op & 255 == 2 => convert.modes[0] + 1,
                _ => return None,
            };
            (*image, 254, conversion)
        }
        _ => return None,
    };
    instruction.op = opcode | (output.op & 0xff00);
    instruction.inputs = [0; 12];
    instruction.inputs[1] = output.modes[0];
    instruction.inputs[2] = conversion;
    Some(instruction)
}

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
                "context" | "shifted_context" => [32, 1, 1, 1],
                "bump" => [2, 1, 1, 1],
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
                "normal_map" => [5, 2, 2, 1],
                "normal" => [2, 1, 1, 1],
                "vector_transform" => [3, 3, 3, 1],
                "camera" | "light_falloff" => [3, 1, 1, 1],
                "radial_tiling" => [4, 2, 1, 1],
                "tangent" => [2, 3, 1, 1],
                "wireframe" => [2, 1, 1, 1],
                "ambient_occlusion" => [2, 2, 2, 129],
                "bevel" => [129, 1, 1, 1],
                "raycast" => [5, 2, 1, 1],
                "attribute" => [5, 1, 1, 1],
                "normal_map_named" => [2, 2, 1, 1],
                "displacement" | "vector_displacement" => [3, 1, 1, 1],
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
                "context" | "shifted_context" => 1 << node.modes[0],
                "image" if node.modes[2] == 1 => 1 << 13,
                "fresnel" | "layer_weight" => (1 << 6) | (1 << 16),
                "normal_map" => (1 << 4) | (1 << 7) | (1 << 16) | (1 << 17) | (0x3f << 22),
                "normal_map_named" => (1 << 4) | (1 << 16) | (0x3f << 22),
                "displacement" | "vector_displacement" => (1 << 4) | (0x3f << 22),
                "vector_transform" => 0x1ff << 22,
                "camera" => 1 << 12,
                "light_falloff" => 1 << 18,
                "tangent" => (1 << 1) | (1 << 4) | (1 << 7) | (0x3f << 22),
                "wireframe" => (1 << 3) | (1 << 6),
                "ambient_occlusion" | "bevel" | "raycast" => (1 << 3) | (1 << 4),
                _ => 0,
            };
        }
        Ok(mask)
    }

    pub fn pack(
        &self,
        textures: &[crate::types::TextureBinding],
        tables: &[u64],
    ) -> Result<Vec<ShaderInstruction>> {
        let packed = self.pack_general(textures, tables)?;
        Ok(match specialize_single_output(&packed) {
            Some(instruction) => vec![instruction],
            None => packed,
        })
    }

    fn pack_general(
        &self,
        textures: &[crate::types::TextureBinding],
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
                let Some(&(data, w, h, channels)) =
                    n.texture.and_then(|id| textures.get(id as usize))
                else {
                    bail!("Missing image binding at shader node {i}")
                };
                ensure!(
                    data != 0 && w > 0 && h > 0,
                    "Invalid shader texture dimensions"
                );
                r.data = data;
                r.width = w;
                r.height = h;
                ensure!((1..=4).contains(&channels), "Invalid texture channel count");
                // Images do not use LUTs; this field carries their pixel stride.
                r.table_len = channels as u32;
            } else if n.op == "attribute" {
                r.data = attribute_key(&n.attribute);
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
pub fn evaluate(graph: &ShaderGraph, contexts: &[[f32; 128]]) -> Result<Vec<[[f32; 4]; 64]>> {
    evaluate_selected(graph, contexts, EVAL_ALL)
}

fn evaluate_selected(
    graph: &ShaderGraph,
    contexts: &[[f32; 128]],
    requested: u32,
) -> Result<Vec<[[f32; 4]; 64]>> {
    evaluate_with_textures_selected(graph, contexts, requested, &[])
}

pub fn evaluate_with_textures(
    graph: &ShaderGraph,
    contexts: &[[f32; 128]],
    textures: &[(&[f32], usize, usize)],
) -> Result<Vec<[[f32; 4]; 64]>> {
    evaluate_with_textures_selected(graph, contexts, EVAL_ALL, textures)
}

fn evaluate_with_textures_selected(
    graph: &ShaderGraph,
    contexts: &[[f32; 128]],
    requested: u32,
    textures: &[(&[f32], usize, usize)],
) -> Result<Vec<[[f32; 4]; 64]>> {
    let graph = graph.optimized()?;
    ensure!(
        !graph.nodes.iter().any(|n| matches!(
            n.op.as_str(),
            "wireframe"
                | "ambient_occlusion"
                | "bevel"
                | "raycast"
                | "attribute"
                | "hair_tangent"
                | "shifted_context"
                | "bump"
                | "vector_bump"
        )),
        "Scene query nodes require renderer geometry; use a render, not evaluate_shader"
    );
    ensure!(
        contexts.len() <= 16_777_216 && contexts.iter().flatten().all(|v| v.is_finite()),
        "Invalid shader contexts"
    );
    let tables: Vec<_> = graph
        .nodes
        .iter()
        .map(|n| n.table.as_ptr() as u64)
        .collect();
    let textures = textures
        .iter()
        .map(|&(data, w, h)| {
            ensure!(
                w > 0
                    && h > 0
                    && w <= i32::MAX as usize
                    && h <= i32::MAX as usize
                    && w.checked_mul(h).and_then(|n| n.checked_mul(4)) == Some(data.len()),
                "Invalid CPU texture dimensions"
            );
            Ok((data.as_ptr() as u64, w as i32, h as i32, 4))
        })
        .collect::<Result<Vec<_>>>()?;
    let instructions = graph.pack(&textures, &tables)?;
    let mut result = Vec::with_capacity(contexts.len());
    for context in contexts {
        let mut output = [[0.0; 4]; 64];
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

    fn run_program(
        program: &[ShaderInstruction],
        context: &[f32; 128],
        group: u32,
    ) -> [[f32; 4]; 64] {
        let mut output = [[-123.0; 4]; 64];
        // SAFETY: callers supply validated programs and keep image bindings
        // alive; context and output have the fixed ABI sizes.
        unsafe {
            vibrt_shader_eval(
                program.as_ptr(),
                program.len() as u32,
                context.as_ptr(),
                output.as_mut_ptr().cast(),
                group,
            );
        }
        output
    }

    #[test]
    fn compact_images_match_rgba_for_filters_extensions_and_alpha() {
        for channels in 1..=4 {
            let mut rgba = Vec::new();
            let mut compact = Vec::new();
            for i in 0..6 {
                let r = i as f32 * 0.25;
                let (g, b) = if channels <= 2 {
                    (r, r)
                } else {
                    (1.0 - r, 2.0 * r)
                };
                let a = if channels % 2 == 0 {
                    i as f32 * 0.2
                } else {
                    1.0
                };
                rgba.extend([r, g, b, a]);
                compact.extend(match channels {
                    1 => vec![r],
                    2 => vec![r, a],
                    3 => vec![r, g, b],
                    _ => vec![r, g, b, a],
                });
            }
            for interpolation in 0..4 {
                for extension in 0..4 {
                    for alpha_flags in 0..8 {
                        let graph: ShaderGraph = serde_json::from_value(json!({
                            "nodes":[{"op":"context"},
                                     {"op":"image","inputs":[0],"texture":0,
                                      "modes":[interpolation,extension,0,alpha_flags & 1],
                                      "params":[0,0,(alpha_flags >> 1) & 1,alpha_flags >> 2]}],
                            "outputs":{"base_color":1}
                        }))
                        .unwrap();
                        let reference = graph
                            .pack(&[(rgba.as_ptr() as u64, 3, 2, 4)], &[0; 2])
                            .unwrap();
                        let packed = [(compact.as_ptr() as u64, 3, 2, channels)];
                        for actual in [
                            graph.pack(&packed, &[0; 2]).unwrap(),
                            graph.pack_general(&packed, &[0; 2]).unwrap(),
                        ] {
                            for uv in [[-1.3, 2.7], [0.0, 1.0], [0.37, 0.61], [1.4, -0.2]] {
                                let mut context = [0.0; 128];
                                context[..4].copy_from_slice(&[uv[0], uv[1], 0.0, 1.0]);
                                assert_eq!(
                                    run_program(&actual, &context, EVAL_ALL),
                                    run_program(&reference, &context, EVAL_ALL)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn specialized_images_match_vm_filtering_alpha_conversion_and_output_groups() {
        for (width, height) in [(3, 5), (4, 2)] {
            let pixels: Vec<f32> = (0..width * height)
                .flat_map(|i| [i as f32 / 11.0, 0.25, 0.5, (i % 4) as f32 / 3.0])
                .collect();
            for interpolation in 0..4 {
                for extension in 0..4 {
                    for alpha_flags in 0..8 {
                        for conversion in 0..5 {
                            let mut nodes = vec![
                                json!({"op":"context","modes":[0,0,0,0]}),
                                json!({"op":"image","inputs":[0],"texture":0,
                                    "modes":[interpolation,extension,0,alpha_flags & 1],
                                    "params":[0,0,(alpha_flags >> 1) & 1,alpha_flags >> 2]}),
                            ];
                            if conversion != 0 {
                                nodes.push(json!({"op":"convert","inputs":[1],
                                                  "modes":[conversion-1,0,0,0]}));
                            }
                            let graph: ShaderGraph = serde_json::from_value(json!({
                                "outputs":{"mix_fac":nodes.len()-1},"nodes":nodes,
                            }))
                            .unwrap();
                            let textures = [(pixels.as_ptr() as u64, width, height, 4)];
                            let tables = vec![0; graph.nodes.len()];
                            let specialized = graph.pack(&textures, &tables).unwrap();
                            let general = graph.pack_general(&textures, &tables).unwrap();
                            assert_eq!(specialized.len(), 1);
                            for uv in [[-1.3, 2.7], [0.0, 1.0], [0.37, 0.61], [1.4, -0.2]] {
                                let mut context = [0.0; 128];
                                context[..4].copy_from_slice(&[uv[0], uv[1], 0.0, 1.0]);
                                for group in
                                    [EVAL_ALL, EVAL_MIX, EVAL_ALPHA, EVAL_BUMP, EVAL_EMISSION]
                                {
                                    assert_eq!(
                                        run_program(&specialized, &context, group),
                                        run_program(&general, &context, group)
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn specialization_keeps_projected_images_and_multiple_outputs_on_vm() {
        let pixels = [1.0f32; 16];
        for (projection, environment) in [(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3)] {
            let graph: ShaderGraph = serde_json::from_value(json!({"nodes":[
                {"op":"context","modes":[0,0,0,0]},
                {"op":"image","inputs":[0],"texture":0,
                 "modes":[0,0,projection,0],"params":[0,environment,0,0]}
            ],"outputs":{"base_color":1}}))
            .unwrap();
            assert_eq!(
                graph
                    .pack(&[(pixels.as_ptr() as u64, 2, 2, 4)], &[0, 0])
                    .unwrap()
                    .len(),
                3
            );
        }
        for node in [
            json!({"op":"constant","params":[0.2,0.4,0.6,0.8]}),
            json!({"op":"context","modes":[0,0,0,0]}),
        ] {
            let mut graph: ShaderGraph = serde_json::from_value(json!({
                "nodes":[node],"outputs":{"alpha":0}
            }))
            .unwrap();
            let mut context = [0.0; 128];
            context[..4].copy_from_slice(&[0.1, 0.3, 0.5, 1.0]);
            let specialized = graph.pack(&[], &[0]).unwrap();
            assert_eq!(specialized.len(), 1);
            for group in [EVAL_ALL, EVAL_ALPHA, EVAL_MIX] {
                assert_eq!(
                    run_program(&specialized, &context, group),
                    run_program(&graph.pack_general(&[], &[0]).unwrap(), &context, group)
                );
            }
            graph.outputs.insert("emission".into(), 0);
            assert_eq!(graph.pack(&[], &[0]).unwrap().len(), 3);
        }
    }

    #[test]
    fn specialized_images_preserve_vm_nonfinite_sanitization() {
        let pixels = [f32::INFINITY, f32::NAN, f32::MAX, 0.5];
        let graph: ShaderGraph = serde_json::from_value(json!({"nodes":[
            {"op":"context","modes":[0,0,0,0]},
            {"op":"image","inputs":[0],"texture":0},
            {"op":"convert","inputs":[1],"modes":[0,0,0,0]}
        ],"outputs":{"base_color":2}}))
        .unwrap();
        let textures = [(pixels.as_ptr() as u64, 1, 1, 4)];
        let specialized = graph.pack(&textures, &[0; 3]).unwrap();
        let general = graph.pack_general(&textures, &[0; 3]).unwrap();
        let mut context = [0.0; 128];
        context[0] = f32::NAN;
        assert_eq!(
            run_program(&specialized, &context, EVAL_ALL),
            run_program(&general, &context, EVAL_ALL)
        );
    }

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
                    .pack(&[(pixels.as_ptr() as u64, width, height, 4)], &[0, 0])
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
                        let mut output = [[0.0; 4]; 64];
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

#[cfg(test)]
mod volume_tests {
    unsafe extern "C" {
        fn vibrt_phase_pdf(kind: u32, p: *const f32, g: f32, cosine: f32) -> f32;
        fn vibrt_phase_sample(kind: u32, p: *const f32, g: f32, u: f32, v: f32, out: *mut f32);
    }
    #[test]
    fn phase_functions_integrate_to_one_and_sample_their_reported_pdf() {
        let parameters = [1.33f32, 0.1, 0.5, 2.0];
        for kind in 0..5 {
            let steps = 131072;
            let mut integral = 0.0f64;
            for i in 0..steps {
                let theta = (i as f64 + 0.5) * std::f64::consts::PI / steps as f64;
                let pdf =
                    unsafe { vibrt_phase_pdf(kind, parameters.as_ptr(), 0.3, theta.cos() as f32) };
                assert!(pdf.is_finite() && pdf >= 0.0, "phase {kind} pdf {pdf}");
                integral +=
                    pdf as f64 * theta.sin() * 2.0 * std::f64::consts::PI.powi(2) / steps as f64;
            }
            assert!(
                (integral - 1.0).abs() < 0.012,
                "phase {kind} integral {integral}"
            );
            for i in 0..1024 {
                let mut result = [0.0; 4];
                unsafe {
                    vibrt_phase_sample(
                        kind,
                        parameters.as_ptr(),
                        0.3,
                        (i as f32 + 0.5) / 1024.0,
                        0.371,
                        result.as_mut_ptr(),
                    );
                }
                assert!(result.iter().all(|x| x.is_finite()));
                assert!(
                    (result[0] * result[0] + result[1] * result[1] + result[2] * result[2] - 1.0)
                        .abs()
                        < 1e-4
                );
                let expected =
                    unsafe { vibrt_phase_pdf(kind, parameters.as_ptr(), 0.3, result[2]) };
                assert!(
                    (expected - result[3]).abs() < 1e-4 * expected.max(1.0),
                    "phase {kind} PDF mismatch"
                );
            }
        }
    }
}
