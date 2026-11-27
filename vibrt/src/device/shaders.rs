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
    pub miss_query: ProgramGroup,
    pub hit_query: ProgramGroup,
    pub curve_query: ProgramGroup,
    pub shader_callable: Option<ProgramGroup>,
    pub query_callable: ProgramGroup,
    pub material_callables: Vec<ProgramGroup>,
    pub hit_radiance: ProgramGroup,
    pub hit_shadow: ProgramGroup,
    pub curve_radiance: ProgramGroup,
    pub curve_shadow: ProgramGroup,
    _module: Module,
    _curve_module: Module,
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
    let ptx_src = pipeline::compile_ptx(requirements)?;
    let ptx_time = t_ptx.elapsed();
    anyhow::ensure!(!progress.cancelled(), "shader compilation cancelled");
    progress.log("Compiling OptiX pipeline...");
    let t_pipeline = Instant::now();
    // --- Pipeline ---
    let pipeline_options = PipelineCompileOptions::new("params")
        .traversable_graph_flags(TraversableGraphFlags::ALLOW_SINGLE_LEVEL_INSTANCING)
        .num_payload_values(2)
        .num_attribute_values(2)
        .uses_primitive_type_flags(PrimitiveTypeFlags::TRIANGLE | PrimitiveTypeFlags::ROUND_LINEAR);

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
    let curve_module = Module::builtin_is(
        ctx,
        &module_opts,
        &pipeline_options,
        PrimitiveType::RoundLinear,
    )?;
    let curve_radiance = ProgramGroup::hitgroup(ctx)
        .closest_hit(&module, "__closesthit__ch")
        .any_hit(&module, "__anyhit__ah")
        .intersection(&curve_module, "")
        .build()?
        .value;
    let curve_shadow = ProgramGroup::hitgroup(ctx)
        .closest_hit(&module, "__closesthit__shadow")
        .any_hit(&module, "__anyhit__ah")
        .intersection(&curve_module, "")
        .build()?
        .value;
    let miss_query = ProgramGroup::miss(ctx, &module, "__miss__query")?.value;
    let hit_query = ProgramGroup::hitgroup(ctx)
        .closest_hit(&module, "__closesthit__query")
        .any_hit(&module, "__anyhit__query")
        .build()?
        .value;
    let curve_query = ProgramGroup::hitgroup(ctx)
        .closest_hit(&module, "__closesthit__query")
        .any_hit(&module, "__anyhit__query")
        .intersection(&curve_module, "")
        .build()?
        .value;
    let query_callable = ProgramGroup::callables(ctx)
        .continuation_callable(&module, "__continuation_callable__query")
        .build()
        .context("query callable")?
        .value;
    let shader_callable = if requirements.shader_ops & (7u64 << 41) != 0 {
        Some(
            ProgramGroup::callables(ctx)
                .continuation_callable(&module, "__continuation_callable__shader")
                .build()
                .context("shader callable")?
                .value,
        )
    } else {
        None
    };
    let mut material_callables = Vec::new();
    if shader_callable.is_some() {
        for name in [
            "__continuation_callable__material",
            "__continuation_callable__emission",
            "__continuation_callable__shadow",
        ] {
            material_callables.push(
                ProgramGroup::callables(ctx)
                    .continuation_callable(&module, name)
                    .build()
                    .with_context(|| name)?
                    .value,
            );
        }
    }
    let mut all_pgs: Vec<&ProgramGroup> = vec![
        &raygen_pg,
        &miss_rg_pg,
        &miss_shadow_pg,
        &hit_rg_pg,
        &hit_shadow_pg,
        &curve_radiance,
        &curve_shadow,
        &miss_query,
        &hit_query,
        &curve_query,
        &query_callable,
    ];
    all_pgs.extend(shader_callable.iter());
    all_pgs.extend(material_callables.iter());
    let pipeline = Pipeline::new(
        ctx,
        &pipeline_options,
        &PipelineLinkOptions { max_trace_depth: 2 },
        &all_pgs,
    )
    .context("pipeline")?
    .value;
    // Node graphs and closure trees change stack use; query the compiled programs.
    pipeline
        .set_stack_size_for_programs(&all_pgs, 2, 1 + 2 * u32::from(shader_callable.is_some()), 2)
        .context("pipeline stack")?;

    anyhow::ensure!(!progress.cancelled(), "shader compilation cancelled");
    Ok(CompiledShaders {
        pipeline,
        raygen: raygen_pg,
        miss_radiance: miss_rg_pg,
        miss_shadow: miss_shadow_pg,
        hit_radiance: hit_rg_pg,
        hit_shadow: hit_shadow_pg,
        curve_radiance,
        curve_shadow,
        miss_query,
        hit_query,
        curve_query,
        shader_callable,
        query_callable,
        material_callables,
        _module: module,
        _curve_module: curve_module,
        ptx_time,
        pipeline_time: t_pipeline.elapsed(),
    })
}
