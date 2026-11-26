//! OptiX shader compilation, using CPU analysis from `material`.
use super::pipeline;
pub use crate::material::{optimize_materials, ShaderRequirements};
use crate::Progress;
use anyhow::{Context, Result};
use optix::*;
use std::time::{Duration, Instant};

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
