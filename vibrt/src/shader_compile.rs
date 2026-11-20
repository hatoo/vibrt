//! Shared shader requirements and OptiX compilation for rendering and cache warmup.
use crate::{
    pipeline,
    scene_format::{PrincipledMaterial, VolumeParams},
    Progress,
};
use anyhow::{Context, Result};
use optix::*;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShaderRequirements {
    pub shader_ops: u64,
    pub max_closures: usize,
    pub shader_stack_size: usize,
    pub closure_types: u32,
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
        fn validate_shader_tree(
            material: &PrincipledMaterial,
            depth: usize,
            shader_stack_size: &mut usize,
            aov_count: usize,
        ) -> Result<usize> {
            anyhow::ensure!(depth <= 8, "Material shader tree exceeds 8 levels");
            if let Some(graph) = &material.shader_graph {
                *shader_stack_size = (*shader_stack_size).max(graph.required_stack_size()?);
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
            }
            if let Some(volume) = &material.volume {
                anyhow::ensure!(
                    volume.phase <= 4 && volume.phase_params.iter().all(|v| v.is_finite()),
                    "Invalid volume phase parameters"
                );
            }
            let leaves = match (&material.left_subtree, &material.secondary) {
                (None, None) => 1,
                (left, right) => {
                    let l = if let Some(c) = left {
                        validate_shader_tree(c, depth + 1, shader_stack_size, aov_count)?
                    } else {
                        1
                    };
                    let r = if let Some(c) = right {
                        validate_shader_tree(c, depth + 1, shader_stack_size, aov_count)?
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
        for material in materials {
            max_closures = max_closures.max(validate_shader_tree(
                material,
                0,
                &mut shader_stack_size,
                aov_count,
            )?);
        }
        if let Some(volume) = &world_volume {
            anyhow::ensure!(
                volume.phase <= 4 && volume.phase_params.iter().all(|v| v.is_finite()),
                "Invalid world volume phase parameters"
            );
        }
        fn shader_operations(material: &PrincipledMaterial) -> u64 {
            let mut flags = 0;
            for graph in material
                .shader_graph
                .iter()
                .chain(material.aov_graphs.iter().map(|a| &a.graph))
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
        let shader_ops = materials
            .iter()
            .fold(0, |mask, m| mask | shader_operations(m));
        fn extended_requirements(m: &PrincipledMaterial) -> Result<(u32, usize)> {
            anyhow::ensure!(
                m.closure_type <= 7 && m.closure_type != 6,
                "Unsupported closure type {}",
                m.closure_type
            );
            let mut kinds = if m.closure_type == 0 {
                0
            } else {
                1 << m.closure_type
            };
            let mut outputs = 32;
            for graph in m
                .shader_graph
                .iter()
                .chain(m.aov_graphs.iter().map(|a| &a.graph))
            {
                if graph
                    .outputs
                    .keys()
                    .any(|name| vibrt_nodes::shader::OUTPUTS[32..].contains(&name.as_str()))
                {
                    outputs = 64;
                }
            }
            for child in [&m.left_subtree, &m.secondary].into_iter().flatten() {
                let (child_kinds, child_outputs) = extended_requirements(child)?;
                kinds |= child_kinds;
                outputs = outputs.max(child_outputs);
            }
            Ok((kinds, outputs))
        }
        let (mut closure_types, mut shader_output_count) = (0, 32);
        for m in materials {
            let (kinds, outputs) = extended_requirements(m)?;
            closure_types |= kinds;
            shader_output_count = shader_output_count.max(outputs);
        }
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
            closure_types,
            shader_output_count,
            has_aovs: aov_count > 0,
            has_extended_phases: extended_phases,
        })
    }

    pub fn scene_queries(self) -> bool {
        self.shader_ops & (7u64 << 41) != 0 || self.closure_types & (1 << 7) != 0
    }
}

// Fields drop in declaration order: programs and pipeline before their module.
pub struct CompiledShaders {
    pub pipeline: Pipeline,
    pub raygen: ProgramGroup,
    pub miss_radiance: ProgramGroup,
    pub miss_shadow: ProgramGroup,
    pub hit_radiance: ProgramGroup,
    pub hit_shadow: ProgramGroup,
    _module: Module,
    pub ptx_time: Duration,
    pub pipeline_time: Duration,
}

pub fn compile(
    ctx: &DeviceContext,
    requirements: ShaderRequirements,
    progress: &mut dyn Progress,
) -> Result<CompiledShaders> {
    anyhow::ensure!(!progress.cancelled(), "shader compilation cancelled");
    progress.log("Preparing shader code...");
    let t_ptx = Instant::now();
    let ptx_src = pipeline::compile_ptx(
        requirements.shader_ops,
        requirements.max_closures,
        requirements.shader_stack_size,
        requirements.closure_types,
        requirements.shader_output_count,
        requirements.has_aovs,
        requirements.has_extended_phases,
    )?;
    let ptx_time = t_ptx.elapsed();
    anyhow::ensure!(!progress.cancelled(), "shader compilation cancelled");
    progress.log("Compiling OptiX pipeline...");
    let t_pipeline = Instant::now();
    // --- Pipeline ---
    let pipeline_options = PipelineCompileOptions::new("params")
        .traversable_graph_flags(TraversableGraphFlags::ALLOW_SINGLE_LEVEL_INSTANCING)
        .num_payload_values(2)
        .num_attribute_values(2)
        .uses_primitive_type_flags(PrimitiveTypeFlags::TRIANGLE);

    let module_opts = ModuleCompileOptions::default();
    let module = Module::new(ctx, &module_opts, &pipeline_options, ptx_src.as_bytes())
        .context("module")?
        .value;

    let raygen_pg = ProgramGroup::raygen(ctx, &module, "__raygen__rg")
        .context("raygen")?
        .value;
    let miss_rg_pg = ProgramGroup::miss(ctx, &module, "__miss__ms")
        .context("miss radiance")?
        .value;
    let miss_shadow_pg = ProgramGroup::miss(ctx, &module, "__miss__shadow")
        .context("miss shadow")?
        .value;
    let hit_rg_pg = ProgramGroup::hitgroup(ctx)
        .closest_hit(&module, "__closesthit__ch")
        .any_hit(&module, "__anyhit__ah")
        .build()
        .context("hitgroup radiance")?
        .value;
    let hit_shadow_pg = ProgramGroup::hitgroup(ctx)
        .closest_hit(&module, "__closesthit__shadow")
        .any_hit(&module, "__anyhit__ah")
        .build()
        .context("hitgroup shadow")?
        .value;
    let all_pgs: Vec<&ProgramGroup> = vec![
        &raygen_pg,
        &miss_rg_pg,
        &miss_shadow_pg,
        &hit_rg_pg,
        &hit_shadow_pg,
    ];
    let pipeline = Pipeline::new(
        ctx,
        &pipeline_options,
        &PipelineLinkOptions { max_trace_depth: 2 },
        &all_pgs,
    )
    .context("pipeline")?
    .value;
    // Node graphs and closure trees change stack use; query the compiled programs.
    pipeline.set_stack_size_for_programs(&all_pgs, 2, 2)?;

    anyhow::ensure!(!progress.cancelled(), "shader compilation cancelled");
    Ok(CompiledShaders {
        pipeline,
        raygen: raygen_pg,
        miss_radiance: miss_rg_pg,
        miss_shadow: miss_shadow_pg,
        hit_radiance: hit_rg_pg,
        hit_shadow: hit_shadow_pg,
        _module: module,
        ptx_time,
        pipeline_time: t_pipeline.elapsed(),
    })
}
