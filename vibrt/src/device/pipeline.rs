//! Assemble and cache device-code compilation.

use anyhow::Result;

pub fn find_optix_include() -> String {
    if let Ok(root) = std::env::var("OPTIX_ROOT") {
        return format!("{root}/include");
    }
    #[cfg(target_os = "windows")]
    {
        let default = r"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\include";
        if std::path::Path::new(default).exists() {
            return default.to_string();
        }
    }
    #[cfg(target_os = "linux")]
    {
        let default = "/usr/local/NVIDIA-OptiX-SDK-9.0.0/include";
        if std::path::Path::new(default).exists() {
            return default.to_string();
        }
    }
    panic!("OptiX SDK not found. Set OPTIX_ROOT.");
}

pub fn compile_ptx(
    shader_ops: u64,
    max_closures: usize,
    shader_stack_size: usize,
    closure_types: u32,
    shader_output_count: usize,
    has_aovs: bool,
    has_extended_phases: bool,
) -> Result<String> {
    // Include expansion also works in Windows checkouts using CRLF endings.
    let mut cu_src = include_str!("../devicecode.cu").replace("\r\n", "\n");
    cu_src = cu_src.replace("#include \"device/math.h\"\n", include_str!("math.h"));
    cu_src = cu_src.replace("#include \"device/volume.h\"\n", include_str!("volume.h"));
    cu_src = cu_src.replace("#include \"device/surface.h\"\n", include_str!("surface.h"));
    cu_src = cu_src.replace(
        "#include \"device/environment.h\"\n",
        include_str!("environment.h"),
    );
    cu_src = cu_src.replace(
        "#include \"device/visibility.h\"\n",
        include_str!("visibility.h"),
    );
    cu_src = cu_src.replace(
        "#include \"device/material.h\"\n",
        include_str!("material.h"),
    );
    cu_src = cu_src.replace(
        "#include \"device/lighting.h\"\n",
        include_str!("lighting.h"),
    );
    cu_src = cu_src.replace(
        "#include \"device/geometry.h\"\n",
        include_str!("geometry.h"),
    );
    cu_src = cu_src.replace(
        "#include \"device/scene_query.h\"\n",
        include_str!("scene_query.h"),
    );
    cu_src = cu_src.replace(
        "#include \"device/programs.h\"\n",
        include_str!("programs.h"),
    );
    cu_src = cu_src.replace(
        "#include \"device/integrator.h\"\n",
        include_str!("integrator.h"),
    );

    let header_src = include_str!("../devicecode.h");
    let opts = cudarc::nvrtc::CompileOptions {
        include_paths: vec![find_optix_include()],
        use_fast_math: Some(true),
        ..Default::default()
    };
    let full_src = format!(
        "#define VIBRT_SHADER_OPS {shader_ops}ull\n#define VIBRT_MAX_CLOSURES {max_closures}\n#define VIBRT_SHADER_STACK_SIZE {shader_stack_size}\n#define VIBRT_CLOSURE_TYPES {closure_types}u\n#define VIBRT_SHADER_OUTPUT_COUNT {shader_output_count}\n#define VIBRT_AOVS {}\n#define VIBRT_EXTENDED_PHASES {}\n// -- inlined devicecode.h --\n{}\n// -- devicecode.cu --\n{}",
        u8::from(has_aovs),
        u8::from(has_extended_phases),
        header_src.replace(
            "#include \"vibrt_nodes_types.h\"",
            vibrt_nodes::DEVICE_TYPES
        ),
        cu_src
            .replace("#include \"devicecode.h\"", "// (inlined above)")
            .replace("#include \"rect_light_sampling.h\"", include_str!("../rect_light_sampling.h"))
            .replace("#include \"vibrt_nodes_core.h\"", vibrt_nodes::DEVICE_CORE)
            .replace("#include \"vibrt_shader.h\"", &vibrt_nodes::DEVICE_SHADER
                .replace("#include \"cycles_nodes.h\"", &vibrt_nodes::DEVICE_CYCLES
                    .replace("#include \"cycles_compat.h\"", vibrt_nodes::DEVICE_CYCLES_COMPAT)))
    );
    use std::hash::{Hash, Hasher};
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    full_src.hash(&mut hash);
    format!("{:?}", opts).hash(&mut hash);
    let key = hash.finish();
    let cache_dir = std::env::temp_dir().join("vibrt-ptx-v1");
    let cache_file = cache_dir.join(format!("{key:016x}.ptx"));
    if let Ok(ptx) = std::fs::read_to_string(&cache_file) {
        if ptx.contains(".version") && ptx.contains("__raygen__rg") {
            return Ok(ptx);
        }
    }
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(&full_src, opts)
        .map_err(|e| anyhow::anyhow!("NVRTC compile failed: {e:?}"))?;
    let ptx = ptx.to_src().to_string();
    if std::fs::create_dir_all(&cache_dir).is_ok() {
        let temporary = cache_file.with_extension(format!("{}.tmp", std::process::id()));
        if std::fs::write(&temporary, &ptx).is_ok() {
            let _ = std::fs::rename(&temporary, &cache_file);
        }
    }
    Ok(ptx)
}
