//! CPU material optimization and shader requirements.
use crate::scene_format::{PrincipledMaterial, VolumeParams};
use anyhow::Result;

/// Prepare the same graphs before render setup and shader-cache warmup.
/// Folding precedes operation/stack/context analysis and GPU LUT uploads.
pub fn optimize_materials(materials: &mut [PrincipledMaterial]) -> Result<()> {
    fn optimize(material: &mut PrincipledMaterial, depth: usize) -> Result<()> {
        anyhow::ensure!(depth <= 8, "Material shader tree exceeds 8 levels");
        if let Some(graph) = &mut material.shader_graph {
            *graph = graph.optimized()?;
        }
        if let Some(graph) = material
            .volume
            .as_mut()
            .and_then(|v| v.emission_graph.as_mut())
        {
            *graph = graph.optimized()?;
        }
        for aov in &mut material.aov_graphs {
            aov.graph = aov.graph.optimized()?;
        }
        for child in [&mut material.left_subtree, &mut material.secondary]
            .into_iter()
            .flatten()
        {
            optimize(child, depth + 1)?;
        }
        Ok(())
    }
    for material in materials {
        optimize(material, 0)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShaderRequirements {
    pub shader_ops: u64,
    pub max_closures: usize,
    pub shader_stack_size: usize,
    pub shader_contexts: u32,
    pub closure_types: u32,
    pub shader_outputs: u64,
    pub shader_output_count: usize,
    pub has_aovs: bool,
    pub has_extended_phases: bool,
}

impl ShaderRequirements {
    pub fn new(
        materials: &[PrincipledMaterial],
        aov_count: usize,
        world_volume: Option<&VolumeParams>,
    ) -> Result<Self> {
        fn validate_volume(v: &VolumeParams, stack: &mut usize, contexts: &mut u32) -> Result<()> {
            anyhow::ensure!(
                v.phase <= 4 && v.phase_params.iter().all(|x| x.is_finite()),
                "Invalid volume phase parameters"
            );
            if let Some(graph) = &v.emission_graph {
                *stack = (*stack).max(graph.required_stack_size()?);
                let mask = graph.required_contexts()?;
                *contexts |= mask;
                anyhow::ensure!(mask & !((1 << 20) | (1 << 21)) == 0,
                    "Homogeneous volume emission supports ray flags; spatial/depth-dependent emission requires heterogeneous volume transport");
                anyhow::ensure!(
                    graph.outputs.len() == 1 && graph.outputs.contains_key("emission"),
                    "Volume graph must output emission"
                );
            }
            Ok(())
        }
        fn validate_shader_tree(
            material: &PrincipledMaterial,
            depth: usize,
            shader_stack_size: &mut usize,
            shader_contexts: &mut u32,
            aov_count: usize,
        ) -> Result<usize> {
            anyhow::ensure!(depth <= 8, "Material shader tree exceeds 8 levels");
            if let Some(graph) = &material.shader_graph {
                *shader_stack_size = (*shader_stack_size).max(graph.required_stack_size()?);
                *shader_contexts |= graph.required_contexts()?;
            }
            for aov in &material.aov_graphs {
                anyhow::ensure!(
                    (aov.index as usize) < aov_count,
                    "AOV index is out of range"
                );
                anyhow::ensure!(
                    aov.graph.outputs.contains_key("base_color"),
                    "AOV graph requires a base_color output"
                );
                *shader_stack_size = (*shader_stack_size).max(aov.graph.required_stack_size()?);
                *shader_contexts |= aov.graph.required_contexts()?;
            }
            if let Some(volume) = &material.volume {
                validate_volume(volume, shader_stack_size, shader_contexts)?;
            }
            let leaves = match (&material.left_subtree, &material.secondary) {
                (None, None) => 1,
                (left, right) => {
                    let l = if let Some(c) = left {
                        validate_shader_tree(
                            c,
                            depth + 1,
                            shader_stack_size,
                            shader_contexts,
                            aov_count,
                        )?
                    } else {
                        1
                    };
                    let r = if let Some(c) = right {
                        validate_shader_tree(
                            c,
                            depth + 1,
                            shader_stack_size,
                            shader_contexts,
                            aov_count,
                        )?
                    } else {
                        1
                    };
                    l + r
                }
            };
            anyhow::ensure!(leaves <= 16, "Material shader tree exceeds 16 BSDF leaves");
            Ok(leaves)
        }
        let mut max_closures = 1;
        let mut shader_stack_size = 1;
        let mut shader_contexts = 0;
        for material in materials {
            max_closures = max_closures.max(validate_shader_tree(
                material,
                0,
                &mut shader_stack_size,
                &mut shader_contexts,
                aov_count,
            )?);
        }
        if let Some(volume) = &world_volume {
            validate_volume(volume, &mut shader_stack_size, &mut shader_contexts)?;
        }
        fn shader_operations(material: &PrincipledMaterial) -> u64 {
            let mut flags = 0;
            for graph in material
                .shader_graph
                .iter()
                .chain(material.aov_graphs.iter().map(|a| &a.graph))
                .chain(
                    material
                        .volume
                        .iter()
                        .filter_map(|v| v.emission_graph.as_ref()),
                )
            {
                for node in &graph.nodes {
                    if let Some(index) = vibrt_nodes::shader::OPS
                        .iter()
                        .position(|(op, _)| *op == node.op)
                    {
                        flags |= 1u64 << index;
                    }
                }
            }
            if let Some(child) = &material.secondary {
                flags |= shader_operations(child);
            }
            if let Some(child) = &material.left_subtree {
                flags |= shader_operations(child);
            }
            flags
        }
        let mut shader_ops = materials
            .iter()
            .fold(0, |mask, m| mask | shader_operations(m));
        if let Some(graph) = world_volume.and_then(|v| v.emission_graph.as_ref()) {
            for node in &graph.nodes {
                if let Some(i) = vibrt_nodes::shader::OPS
                    .iter()
                    .position(|(op, _)| *op == node.op)
                {
                    shader_ops |= 1u64 << i;
                }
            }
        }
        fn graph_outputs(graph: &vibrt_nodes::shader::ShaderGraph) -> u64 {
            vibrt_nodes::shader::OUTPUTS
                .iter()
                .enumerate()
                .filter(|(_, name)| graph.outputs.contains_key(**name))
                .fold(0, |mask, (slot, _)| mask | (1u64 << slot))
        }
        fn extended_requirements(m: &PrincipledMaterial) -> Result<(u32, u64)> {
            anyhow::ensure!(
                m.surface_model <= 8 && m.distribution <= 2,
                "Invalid Cycles surface model/distribution"
            );
            anyhow::ensure!(
                m.closure_type <= 8,
                "Unsupported closure type {}",
                m.closure_type
            );
            let mut kinds = if m.closure_type == 0 {
                0
            } else {
                1 << m.closure_type
            };
            let mut outputs = 0;
            anyhow::ensure!(m.sss_method <= 3, "Unsupported subsurface method");
            if m.sss_method != 0 {
                kinds |= 1 << 7;
            }
            for graph in m
                .shader_graph
                .iter()
                .chain(m.aov_graphs.iter().map(|a| &a.graph))
                .chain(m.volume.iter().filter_map(|v| v.emission_graph.as_ref()))
            {
                outputs |= graph_outputs(graph);
            }
            for child in [&m.left_subtree, &m.secondary].into_iter().flatten() {
                let (child_kinds, child_outputs) = extended_requirements(child)?;
                kinds |= child_kinds;
                outputs |= child_outputs;
            }
            Ok((kinds, outputs))
        }
        let (mut closure_types, mut shader_outputs) = (0, 0);
        for m in materials {
            let (kinds, outputs) = extended_requirements(m)?;
            closure_types |= kinds;
            shader_outputs |= outputs;
        }
        if let Some(graph) = world_volume.and_then(|v| v.emission_graph.as_ref()) {
            shader_outputs |= graph_outputs(graph);
        }
        // Keep the existing 32/64-slot buckets, including the implicit bump
        // reads at 27/28 and Chiang/Huang hair reads through 57. Static SSS parameters
        // live in PrincipledGpu and do not require upper graph output slots.
        let shader_output_count =
            if shader_outputs >> 32 != 0 || closure_types & ((1 << 5) | (1 << 6)) != 0 {
                64
            } else {
                32
            };
        // Bucket capacities to keep a small set of cached kernels across scene edits.
        fn extended_volume(m: &PrincipledMaterial) -> bool {
            m.volume.as_ref().is_some_and(|v| v.phase != 0)
                || [&m.left_subtree, &m.secondary]
                    .into_iter()
                    .flatten()
                    .any(|m| extended_volume(m))
        }
        let extended_phases = world_volume.as_ref().is_some_and(|v| v.phase != 0)
            || materials.iter().any(extended_volume);

        Ok(Self {
            shader_ops,
            max_closures,
            shader_stack_size: shader_stack_size.max(4).next_power_of_two(),
            shader_contexts,
            closure_types,
            shader_outputs,
            shader_output_count,
            has_aovs: aov_count > 0,
            has_extended_phases: extended_phases,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn volume_emission_contributes_shader_requirements_and_rejects_spatial_context() {
        let graph = json!({"nodes":[
            {"op":"context","modes":[20,0,0,0]},
            {"op":"separate","inputs":[0],"modes":[0,0,0,0]}
        ],"outputs":{"emission":1}});
        let mut volume: VolumeParams =
            serde_json::from_value(json!({"emission_graph":graph})).unwrap();
        let world = ShaderRequirements::new(&[], 0, Some(&volume)).unwrap();
        let material: PrincipledMaterial = serde_json::from_value(json!({
            "base_color":[0,0,0], "volume":{"emission_graph":graph}
        }))
        .unwrap();
        let bounded = ShaderRequirements::new(&[material], 0, None).unwrap();
        assert_ne!(world.shader_ops, 0);
        assert_eq!(world.shader_ops, bounded.shader_ops);
        assert_eq!(world.shader_contexts, 1 << 20);
        assert_eq!(bounded.shader_contexts, world.shader_contexts);
        volume.emission_graph.as_mut().unwrap().nodes[0].modes[0] = 3;
        assert!(ShaderRequirements::new(&[], 0, Some(&volume)).is_err());
    }

    #[test]
    fn principled_subsurface_enables_closure_without_a_standalone_node() {
        let material: PrincipledMaterial = serde_json::from_value(json!({
            "base_color":[0.2,0.4,0.7],"sss_weight":1,"sss_method":2
        }))
        .unwrap();
        let requirements = ShaderRequirements::new(&[material], 0, None).unwrap();
        assert_ne!(requirements.closure_types & (1 << 7), 0);
        assert_eq!(requirements.shader_output_count, 32);
        assert_eq!(requirements.shader_outputs, 0);
    }

    #[test]
    fn output_requirements_cover_nested_materials_aovs_and_volumes() {
        let graph = |name: &str| {
            json!({"nodes":[{"op":"constant","params":[1,1,1,1]}],
                   "outputs":{name:0}})
        };
        let material: PrincipledMaterial = serde_json::from_value(json!({
            "base_color":[1,1,1], "shader_graph":graph("normal"),
            "left_subtree":{
                "base_color":[1,1,1], "shader_graph":graph("sss_scale")
            },
            "secondary":{
                "base_color":[1,1,1], "shader_graph":graph("coat_tint"),
                "aov_graphs":[{"index":0,"graph":{
                    "nodes":[{"op":"constant","params":[1,1,1,1]}],
                    "outputs":{"base_color":0,"volume_emission":0}
                }}],
                "volume":{"emission_graph":graph("emission")}
            }
        }))
        .unwrap();
        let requirements = ShaderRequirements::new(&[material], 1, None).unwrap();
        assert_eq!(
            requirements.shader_outputs,
            1 | (1 << 5) | (1 << 24) | (1 << 37) | (1 << 41) | (1 << 63)
        );
        assert_eq!(requirements.shader_output_count, 64);
        let volume = serde_json::from_value(json!({"emission_graph":graph("emission")})).unwrap();
        let world = ShaderRequirements::new(&[], 0, Some(&volume)).unwrap();
        assert_eq!(world.shader_outputs, 1 << 5);
        assert_eq!(world.shader_output_count, 32);
    }

    #[test]
    fn output_only_edits_change_the_shader_cache_key() {
        let mut material: PrincipledMaterial = serde_json::from_value(json!({
            "base_color":[1,1,1], "shader_graph":{
                "nodes":[{"op":"constant","params":[1,1,1,1]}],
                "outputs":{"base_color":0}
            }
        }))
        .unwrap();
        let before = ShaderRequirements::new(std::slice::from_ref(&material), 0, None).unwrap();
        let outputs = &mut material.shader_graph.as_mut().unwrap().outputs;
        outputs.remove("base_color");
        outputs.insert("roughness".into(), 0);
        let after = ShaderRequirements::new(&[material], 0, None).unwrap();
        assert_eq!(before.shader_ops, after.shader_ops);
        assert_eq!(before.shader_stack_size, after.shader_stack_size);
        assert_eq!(before.shader_contexts, after.shader_contexts);
        assert_eq!(before.shader_output_count, after.shader_output_count);
        assert_eq!(before.shader_outputs, 1);
        assert_eq!(after.shader_outputs, 1 << 1);
        assert_ne!(before, after);
    }

    #[test]
    fn output_capacity_covers_all_writes_and_implicit_hair_reads() {
        for (slot, name) in vibrt_nodes::shader::OUTPUTS.iter().enumerate() {
            let material: PrincipledMaterial = serde_json::from_value(json!({
                "base_color":[1,1,1], "shader_graph":{
                    "nodes":[{"op":"constant","params":[1,1,1,1]}],
                    "outputs":{*name:0}
                }
            }))
            .unwrap();
            let requirements = ShaderRequirements::new(&[material], 0, None).unwrap();
            assert_eq!(requirements.shader_outputs, 1 << slot);
            assert_eq!(
                requirements.shader_output_count,
                if slot < 32 { 32 } else { 64 }
            );
        }
        for kind in [5, 6] {
            let hair = serde_json::from_value(json!({
                "base_color":[1,1,1], "closure_type":kind
            }))
            .unwrap();
            assert_eq!(
                ShaderRequirements::new(&[hair], 0, None)
                    .unwrap()
                    .shader_output_count,
                64
            );
        }
    }

    #[test]
    fn context_requirements_cover_nested_materials_aovs_and_implicit_reads() {
        let context_graph = |slot| {
            json!({"nodes":[{"op":"context","modes":[slot,0,0,0]}],
                   "outputs":{"base_color":0}})
        };
        let material: PrincipledMaterial = serde_json::from_value(json!({
            "base_color":[1,1,1], "shader_graph":context_graph(0),
            "left_subtree":{
                "base_color":[1,1,1], "shader_graph":context_graph(31)
            },
            "secondary":{
                "base_color":[1,1,1],
                "shader_graph":{"nodes":[
                    {"op":"constant","params":[0,0,1,1]},
                    {"op":"fresnel","inputs":[0,0]}
                ],"outputs":{"base_color":1}},
                "aov_graphs":[{"index":0,"graph":context_graph(12)}],
                "volume":{"emission_graph":{
                    "nodes":[{"op":"context","modes":[21,0,0,0]}],
                    "outputs":{"emission":0}
                }}
            }
        }))
        .unwrap();
        let requirements = ShaderRequirements::new(&[material], 1, None).unwrap();
        assert_eq!(
            requirements.shader_contexts,
            (1 << 0) | (1 << 6) | (1 << 12) | (1 << 16) | (1 << 21) | (1 << 31)
        );
    }

    #[test]
    fn context_only_edits_change_the_shader_cache_key() {
        let mut material: PrincipledMaterial = serde_json::from_value(json!({
            "base_color":[1,1,1], "shader_graph":{
                "nodes":[{"op":"context","modes":[0,0,0,0]}],
                "outputs":{"base_color":0}
            }
        }))
        .unwrap();
        let before = ShaderRequirements::new(std::slice::from_ref(&material), 0, None).unwrap();
        material.shader_graph.as_mut().unwrap().nodes[0].modes[0] = 31;
        let after = ShaderRequirements::new(&[material], 0, None).unwrap();
        assert_eq!(before.shader_ops, after.shader_ops);
        assert_eq!(before.shader_stack_size, after.shader_stack_size);
        assert_eq!(before.shader_contexts, 1);
        assert_eq!(after.shader_contexts, 1 << 31);
        assert_ne!(before, after);
        assert_eq!(
            ShaderRequirements::new(&[], 0, None)
                .unwrap()
                .shader_contexts,
            0
        );
    }

    #[test]
    fn optimization_reaches_closure_children_and_aovs_before_requirements() {
        let graph = json!({"nodes":[
            {"op":"constant","params":[2.0,2.0,2.0,1.0]},
            {"op":"constant","params":[3.0,3.0,3.0,1.0]},
            {"op":"math","inputs":[0,1,0],"modes":[2,0,0,0]}
        ],"outputs":{"base_color":2}});
        let leaf = json!({
            "base_color":[1.0,1.0,1.0], "shader_graph":graph,
            "aov_graphs":[{"index":0,"graph":graph}]
        });
        let mut materials: Vec<PrincipledMaterial> = serde_json::from_value(json!([{
            "base_color":[1.0,1.0,1.0],"shader_graph":graph,
            "left_subtree":leaf,
            "secondary":{
                "base_color":[1.0,1.0,1.0],"shader_graph":graph,
                "secondary":leaf
            }
        }]))
        .unwrap();
        let before = ShaderRequirements::new(&materials, 1, None).unwrap();
        assert_ne!(before.shader_ops & (1 << 3), 0);
        optimize_materials(&mut materials).unwrap();
        let after = ShaderRequirements::new(&materials, 1, None).unwrap();
        assert_eq!(after.shader_ops, 1); // Only constant instructions remain.
        assert_eq!(after.max_closures, before.max_closures);
        assert!(after.has_aovs);
        let aov = &materials[0].left_subtree.as_ref().unwrap().aov_graphs[0].graph;
        assert_eq!(aov.nodes.len(), 1);
        assert_eq!(aov.nodes[0].params, [6.0, 6.0, 6.0, 1.0]);
        let nested = materials[0]
            .secondary
            .as_ref()
            .unwrap()
            .secondary
            .as_ref()
            .unwrap();
        assert_eq!(nested.shader_graph.as_ref().unwrap().nodes.len(), 1);
        assert_eq!(nested.aov_graphs[0].graph.nodes.len(), 1);
    }
}
