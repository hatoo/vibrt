//! Constant propagation and common expressions using the shared C++ shader VM.
use super::{
    vibrt_shader_eval, ShaderGraph, ShaderInstruction, ShaderNode, EVAL_ALL, OPS, OUTPUTS,
};
use anyhow::Result;
use std::collections::HashMap;

// An explicit allowlist is intentional: some nodes read shading context or
// renderer services even when every socket is constant. New operations must
// be checked against native/shader.h before they can be evaluated here.
fn is_pure(op: &str) -> bool {
    matches!(
        op,
        "convert"
            | "math"
            | "vector_math"
            | "mix"
            | "invert"
            | "hsv"
            | "gamma"
            | "bright_contrast"
            | "combine"
            | "separate"
            | "clamp"
            | "map_range"
            | "mapping"
            | "vector_rotate"
            | "noise"
            | "voronoi"
            | "wave"
            | "checker"
            | "brick"
            | "gradient"
            | "magic"
            | "white_noise"
            | "gabor"
            | "curve"
            | "ramp"
            | "blackbody"
            | "wavelength"
            | "normal"
            | "squeeze"
            | "radial_tiling"
    )
}

// These operations read immutable inputs/resources for one shading point.
// Sharing an expression does not make it constant: context and derivative
// offsets are still evaluated on every invocation. Keep traced scene queries
// separate so their sampling/evaluation order is unchanged.
fn is_shareable(op: &str) -> bool {
    is_pure(op)
        || matches!(
            op,
            "constant"
                | "context"
                | "shifted_context"
                | "image"
                | "attribute"
                | "fresnel"
                | "layer_weight"
                | "normal_map"
                | "normal_map_named"
                | "vector_transform"
                | "camera"
                | "light_falloff"
                | "tangent"
                | "hair_tangent"
                | "bump"
                | "vector_bump"
                | "displacement"
                | "vector_displacement"
        )
}

#[derive(PartialEq, Eq, Hash)]
struct ExpressionKey {
    op: String,
    inputs: Vec<u32>,
    modes: [u32; 4],
    params: [u32; 4],
    texture: Option<u32>,
    table: Vec<u32>,
    attribute: String,
}

impl From<&ShaderNode> for ExpressionKey {
    fn from(node: &ShaderNode) -> Self {
        Self {
            op: node.op.clone(),
            inputs: node.inputs.clone(),
            modes: node.modes,
            // Bit keys preserve signed zero and alpha, including LUT entries.
            params: node.params.map(f32::to_bits),
            texture: node.texture,
            table: node.table.iter().map(|v| v.to_bits()).collect(),
            attribute: node.attribute.clone(),
        }
    }
}

fn constant_value(node: &ShaderNode, values: &[Option<[f32; 4]>]) -> Option<[f32; 4]> {
    if node.op == "constant" {
        return Some(node.params);
    }
    if !is_pure(&node.op) {
        return None;
    }

    // At most 12 operand loads, one operation and one output. Run the actual
    // VM, including its float4/alpha rules and per-instruction non-finite
    // cleanup, instead of reproducing numerical behavior in Rust.
    let mut program = [ShaderInstruction::default(); 14];
    let slot = node.inputs.len();
    for (i, &input) in node.inputs.iter().enumerate() {
        program[i].op = (EVAL_ALL << 8) | ((i as u32) << 16);
        program[i].params = values[input as usize]?;
    }
    let opcode = OPS.iter().position(|(op, _)| *op == node.op)? as u32;
    program[slot] = ShaderInstruction {
        op: opcode | (EVAL_ALL << 8) | ((slot as u32) << 16),
        modes: node.modes,
        params: node.params,
        data: node.table.as_ptr() as u64,
        table_len: node.table.len() as u32,
        ..Default::default()
    };
    for i in 0..slot {
        program[slot].inputs[i] = i as u32;
    }
    program[slot + 1].op = 255 | (EVAL_ALL << 8);
    program[slot + 1].inputs[0] = slot as u32;
    let context = [0.0; 128];
    let mut output = [[0.0; 4]; OUTPUTS.len()];
    // SAFETY: optimized() validates the graph before calling this helper.
    // The allowlist excludes all context, image and scene-service reads;
    // LUT storage is borrowed from the validated node throughout this call.
    // Every operand slot is initialized and all ABI buffers have fixed sizes.
    unsafe {
        vibrt_shader_eval(
            program.as_ptr(),
            (slot + 2) as u32,
            context.as_ptr(),
            output.as_mut_ptr().cast(),
            EVAL_ALL,
        );
    }
    Some(output[0])
}

impl ShaderGraph {
    /// Fold pure constant subgraphs, remove unreachable nodes, and share equal
    /// expressions. Validate the original graph first, including unused nodes.
    /// Images, scene queries and implicit context reads remain dynamic.
    pub fn optimized(&self) -> Result<Self> {
        self.validate()?;
        let mut values = Vec::with_capacity(self.nodes.len());
        let mut nodes = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let value = constant_value(node, &values);
            nodes.push(match value {
                Some(params) => ShaderNode {
                    op: "constant".into(),
                    inputs: Vec::new(),
                    modes: [0; 4],
                    params,
                    texture: None,
                    table: Vec::new(),
                    attribute: String::new(),
                },
                None => node.clone(),
            });
            values.push(value);
        }

        // Traverse the folded graph: the operands of a folded result are no
        // longer live unless another output or dynamic node still uses them.
        let mut live = vec![false; nodes.len()];
        for &id in self.outputs.values() {
            live[id as usize] = true;
        }
        for i in (0..nodes.len()).rev() {
            if live[i] {
                for &input in &nodes[i].inputs {
                    live[input as usize] = true;
                }
            }
        }
        let mut remap = vec![0; nodes.len()];
        let mut expressions = HashMap::new();
        let mut compact = Vec::new();
        for (i, mut node) in nodes.into_iter().enumerate() {
            if !live[i] {
                continue;
            }
            for input in &mut node.inputs {
                *input = remap[*input as usize];
            }
            if is_shareable(&node.op) {
                let key = ExpressionKey::from(&node);
                if let Some(&id) = expressions.get(&key) {
                    remap[i] = id;
                    continue;
                }
                expressions.insert(key, compact.len() as u32);
            }
            remap[i] = compact.len() as u32;
            compact.push(node);
        }
        Ok(Self {
            nodes: compact,
            outputs: self
                .outputs
                .iter()
                .map(|(name, &id)| (name.clone(), remap[id as usize]))
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn graph(value: serde_json::Value) -> ShaderGraph {
        serde_json::from_value(value).unwrap()
    }

    // Deliberately bypass optimized()/evaluate(): the original VM is the
    // numerical oracle, so a broken fold cannot be used on both test sides.
    fn raw_eval(g: &ShaderGraph, context: &[f32; 128], group: u32) -> [[f32; 4]; 64] {
        let tables: Vec<_> = g.nodes.iter().map(|n| n.table.as_ptr() as u64).collect();
        let program = g.pack(&[], &tables).unwrap();
        let mut output = [[-123.0; 4]; 64];
        // SAFETY: pack validates all indices; LUT owners and ABI arrays live
        // through this synchronous call. These tests do not bind textures.
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

    fn assert_bits_equal(a: [[f32; 4]; 64], b: [[f32; 4]; 64]) {
        assert_eq!(
            a.map(|v| v.map(f32::to_bits)),
            b.map(|v| v.map(f32::to_bits))
        );
    }

    #[test]
    fn shares_dynamic_expressions_across_outputs_and_evaluation_groups() {
        let g = graph(json!({"nodes":[
            {"op":"context","modes":[0,0,0,0]},
            {"op":"context","modes":[0,0,0,0]},
            {"op":"math","inputs":[0,0,0],"modes":[2,0,0,0]},
            {"op":"math","inputs":[1,1,1],"modes":[2,0,0,0]},
            {"op":"mix","inputs":[0,2,3],"modes":[0,1,0,0]}
        ],"outputs":{"base_color":4,"emission":2,"bump_height":3,"mix_fac":3}}));
        let shared = g.optimized().unwrap();
        assert_eq!(shared.nodes.len(), 3);
        assert_eq!(shared.outputs["emission"], shared.outputs["bump_height"]);
        for x in [-1.5, 0.0, 0.25, 3.0] {
            let mut context = [0.0; 128];
            context[0] = x;
            context[3] = 0.37;
            for group in [
                EVAL_ALL,
                super::super::EVAL_MIX,
                super::super::EVAL_BUMP,
                super::super::EVAL_EMISSION,
                super::super::EVAL_ALPHA,
            ] {
                assert_bits_equal(
                    raw_eval(&g, &context, group),
                    raw_eval(&shared, &context, group),
                );
            }
        }
        assert_eq!(
            serde_json::to_value(shared.optimized().unwrap()).unwrap(),
            serde_json::to_value(shared).unwrap()
        );
    }

    #[test]
    fn expression_keys_preserve_resources_modes_derivatives_and_float_bits() {
        for (a, b) in [
            (
                json!({"op":"image","inputs":[0],"texture":0}),
                json!({"op":"image","inputs":[0],"texture":1}),
            ),
            (
                json!({"op":"image","inputs":[0],"texture":0}),
                json!({"op":"image","inputs":[0],"texture":0,"modes":[1,0,0,0]}),
            ),
            (
                json!({"op":"shifted_context","params":[0.001,0.0,0.0,0.0]}),
                json!({"op":"shifted_context","params":[0.0,0.001,0.0,0.0]}),
            ),
            (
                json!({"op":"attribute","attribute":"UVMap"}),
                json!({"op":"attribute","attribute":"UVMap.001"}),
            ),
            (
                json!({"op":"ramp","inputs":[0],"table":[0.0,0.2,0.4,0.0,1.0,0.7,0.8,1.0]}),
                json!({"op":"ramp","inputs":[0],"table":[0.0,0.2,0.4,-0.0,1.0,0.7,0.8,1.0]}),
            ),
            (
                json!({"op":"ramp","inputs":[0],"params":[0.0,0.0,0.0,0.0],"table":[0.0,0.2,0.4,0.0,1.0,0.7,0.8,1.0]}),
                json!({"op":"ramp","inputs":[0],"params":[-0.0,0.0,0.0,0.0],"table":[0.0,0.2,0.4,0.0,1.0,0.7,0.8,1.0]}),
            ),
        ] {
            let g = graph(json!({"nodes":[{"op":"context"}, a, b],
                                "outputs":{"base_color":1,"emission":2}}));
            let shared = g.optimized().unwrap();
            assert_ne!(shared.outputs["base_color"], shared.outputs["emission"]);
            assert_eq!(
                g.required_contexts().unwrap(),
                shared.required_contexts().unwrap()
            );
        }
    }

    #[test]
    fn leaves_traced_queries_separate() {
        for op in ["ambient_occlusion", "bevel", "raycast"] {
            let arity = OPS.iter().find(|(name, _)| *name == op).unwrap().1;
            let node = json!({"op":op,"inputs":vec![0;arity]});
            let g = graph(json!({"nodes":[{"op":"context"}, node, node],
                                "outputs":{"base_color":1,"emission":2}}));
            let shared = g.optimized().unwrap();
            assert_eq!(shared.nodes.len(), 3);
            assert_ne!(shared.outputs["base_color"], shared.outputs["emission"]);
        }
    }

    fn pure_graph(op: &str, modes: [u32; 4]) -> ShaderGraph {
        let arity = OPS.iter().find(|(name, _)| *name == op).unwrap().1;
        let mut nodes: Vec<_> = (0..arity)
            .map(|i| {
                let v = (i + 1) as f32 * 0.17;
                json!({"op":"constant", "params":[v, v * 2.0, -v, 0.37]})
            })
            .collect();
        nodes.push(json!({
            "op":op, "inputs":(0..arity).collect::<Vec<_>>(), "modes":modes,
            "params":[0.0,1.0,0.5,2.0],
            "table":if matches!(op, "curve" | "ramp") {
                vec![-2.0,0.3,4.0,0.25, 3.0,0.7,-1.0,0.75]
            } else { vec![] }
        }));
        graph(json!({"nodes":nodes,"outputs":{"base_color":arity}}))
    }

    fn assert_folds_to_vm_result(g: &ShaderGraph) {
        let folded = g.optimized().unwrap();
        assert_eq!(folded.nodes.len(), 1);
        assert_eq!(folded.nodes[0].op, "constant");
        assert!(folded.nodes[0].table.is_empty());
        assert_eq!(folded.pack(&[], &[0]).unwrap().len(), 1);
        assert_bits_equal(
            raw_eval(g, &[0.0; 128], EVAL_ALL),
            raw_eval(&folded, &[0.0; 128], EVAL_ALL),
        );
    }

    #[test]
    fn folds_every_pure_operation_using_the_vm() {
        for &(op, _) in &OPS {
            if !is_pure(op) {
                continue;
            }
            let modes = if matches!(op, "noise" | "voronoi" | "white_noise") {
                [3, 0, 0, 0]
            } else {
                [0; 4]
            };
            assert_folds_to_vm_result(&pure_graph(op, modes));
        }
        for mode in 0..41 {
            for clamp in 0..2 {
                assert_folds_to_vm_result(&pure_graph("math", [mode, clamp, 0, 0]));
            }
        }
        for mode in 0..30 {
            for output in 0..2 {
                assert_folds_to_vm_result(&pure_graph("vector_math", [mode, output, 0, 0]));
            }
        }
        for mode in 0..19 {
            for data in 0..3 {
                assert_folds_to_vm_result(&pure_graph("mix", [mode, data, 1, 1]));
            }
        }
    }

    #[test]
    fn folds_chains_and_remaps_shared_dynamic_multi_output_graphs() {
        let g = graph(json!({"nodes":[
            {"op":"constant","params":[2.0,2.0,2.0,1.0]},
            {"op":"constant","params":[3.0,3.0,3.0,1.0]},
            {"op":"math","inputs":[0,1,0],"modes":[2,0,0,0]},
            {"op":"context","modes":[0,0,0,0]},
            {"op":"math","inputs":[2,3,0]},
            {"op":"constant","params":[6.0,6.0,6.0,1.0]},
            {"op":"context","modes":[1,0,0,0]}
        ],"outputs":{"base_color":4,"alpha":2,"roughness":0,"emission_strength":5}}));
        let folded = g.optimized().unwrap();
        assert_eq!(folded.nodes.len(), 4); // 2, 6, context and dynamic Add.
        assert_eq!(folded.outputs["alpha"], folded.outputs["emission_strength"]);
        assert_eq!(folded.required_contexts().unwrap(), 1);
        for value in [-3.0, 0.0, 0.5, 2.0] {
            let mut context = [0.0; 128];
            context[0] = value;
            for group in [
                EVAL_ALL,
                super::super::EVAL_ALPHA,
                super::super::EVAL_EMISSION,
            ] {
                assert_bits_equal(
                    raw_eval(&g, &context, group),
                    raw_eval(&folded, &context, group),
                );
            }
        }
        assert_eq!(
            serde_json::to_value(&folded).unwrap(),
            serde_json::to_value(folded.optimized().unwrap()).unwrap()
        );

        let chain = graph(json!({"nodes":[
            {"op":"constant","params":[2.0,2.0,2.0,1.0]},
            {"op":"math","inputs":[0,0,0],"modes":[2,0,0,0]},
            {"op":"math","inputs":[1,1,0],"modes":[0,0,0,0]}
        ],"outputs":{"base_color":2}}));
        assert_folds_to_vm_result(&chain);
        assert_eq!(
            chain.optimized().unwrap().nodes[0].params,
            [8.0, 8.0, 8.0, 1.0]
        );
    }

    #[test]
    fn preserves_implicit_context_and_resource_dependencies() {
        for &(op, arity) in &OPS {
            if op == "constant" || is_pure(op) {
                continue;
            }
            let g = graph(json!({"nodes":[
                {"op":"constant","params":[0.5,0.5,0.5,1.0]},
                {"op":op,"inputs":vec![0;arity],"texture":0,"attribute":"fixture"}
            ],"outputs":{"base_color":1}}));
            let folded = g.optimized().unwrap();
            assert_eq!(folded.nodes.last().unwrap().op, op);
            assert_eq!(
                folded.required_contexts().unwrap(),
                g.required_contexts().unwrap()
            );
        }

        let fresnel = graph(json!({"nodes":[
            {"op":"constant","params":[1.45,1.45,1.45,1.0]},
            {"op":"constant","params":[0.0,0.0,1.0,1.0]},
            {"op":"fresnel","inputs":[0,1]}
        ],"outputs":{"base_color":2}}));
        let folded = fresnel.optimized().unwrap();
        let mut a = [0.0; 128];
        a[26] = 1.0;
        let mut b = a;
        b[26] = 0.25;
        assert_ne!(
            raw_eval(&folded, &a, EVAL_ALL)[0],
            raw_eval(&folded, &b, EVAL_ALL)[0]
        );
        assert_bits_equal(
            raw_eval(&fresnel, &b, EVAL_ALL),
            raw_eval(&folded, &b, EVAL_ALL),
        );
    }

    #[test]
    fn preserves_hdr_alpha_signed_zero_and_nonfinite_cleanup() {
        let g = graph(json!({"nodes":[
            {"op":"constant","params":[-2.0,4.0,0.0,0.25]},
            {"op":"constant","params":[-2.0,4.0,-0.0,0.25]},
            {"op":"constant","params":[-2.0,4.0,0.0,0.75]}
        ],"outputs":{"base_color":0,"emission":1,"normal":2}}));
        let folded = g.optimized().unwrap();
        assert_eq!(folded.nodes.len(), 3);
        assert_bits_equal(
            raw_eval(&g, &[0.0; 128], EVAL_ALL),
            raw_eval(&folded, &[0.0; 128], EVAL_ALL),
        );

        let mut overflow = pure_graph("math", [2, 0, 0, 0]);
        overflow.nodes[0].params = [f32::MAX; 4];
        overflow.nodes[1].params = [f32::MAX; 4];
        assert_folds_to_vm_result(&overflow);
        assert_eq!(
            overflow.optimized().unwrap().nodes[0].params,
            [0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn rejects_invalid_even_when_unreachable_or_foldable() {
        let mut g = pure_graph("math", [0; 4]);
        let id = g.nodes.len() as u32;
        let mut bad = g.nodes.last().unwrap().clone();
        bad.inputs[0] = id; // Unreachable self-reference must not be pruned away.
        g.nodes.push(bad);
        assert!(g.optimized().is_err());
        let mut g = pure_graph("ramp", [0; 4]);
        g.nodes.last_mut().unwrap().table.clear();
        assert!(g.optimized().is_err());
        let mut g = pure_graph("math", [41, 0, 0, 0]);
        assert!(g.optimized().is_err());
        g.nodes.last_mut().unwrap().modes[0] = 0;
        g.nodes[0].params[0] = f32::NAN;
        assert!(g.optimized().is_err());
    }
}
