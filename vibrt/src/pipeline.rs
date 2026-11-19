//! Device-code compilation + precomputed lookup tables.

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
) -> Result<String> {
    let cu_src = include_str!("devicecode.cu");
    let header_src = include_str!("devicecode.h");
    let opts = cudarc::nvrtc::CompileOptions {
        include_paths: vec![find_optix_include()],
        use_fast_math: Some(true),
        ..Default::default()
    };
    let full_src = format!(
        "#define VIBRT_SHADER_OPS {shader_ops}ull\n#define VIBRT_MAX_CLOSURES {max_closures}\n#define VIBRT_SHADER_STACK_SIZE {shader_stack_size}\n// -- inlined devicecode.h --\n{}\n// -- devicecode.cu --\n{}",
        header_src.replace(
            "#include \"vibrt_nodes_types.h\"",
            vibrt_nodes::DEVICE_TYPES
        ),
        cu_src
            .replace("#include \"devicecode.h\"", "// (inlined above)")
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

// The integration depends only on the BRDF, never on the scene or GPU.
// Build it once and embed the exact float bits, including for the first render.
include!(concat!(env!("OUT_DIR"), "/ggx_energy_lut.rs"));

/// Build marginal + conditional CDFs for envmap importance sampling (RGB data).
pub fn build_envmap_cdf(rgb: &[f32], width: u32, height: u32) -> (Vec<f32>, Vec<f32>, f32) {
    let w = width as usize;
    let h = height as usize;
    let mut cond = vec![0.0f32; h * (w + 1)];
    let mut row_int = vec![0.0f32; h];
    for y in 0..h {
        let sin_t = (std::f32::consts::PI * (y as f32 + 0.5) / h as f32).sin();
        let ro = y * (w + 1);
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let lum = 0.2126 * rgb[idx] + 0.7152 * rgb[idx + 1] + 0.0722 * rgb[idx + 2];
            cond[ro + x + 1] = cond[ro + x] + lum * sin_t;
        }
        row_int[y] = cond[ro + w];
        if row_int[y] > 0.0 {
            let inv = 1.0 / row_int[y];
            for x in 1..=w {
                cond[ro + x] *= inv;
            }
        }
    }
    let mut marg = vec![0.0f32; h + 1];
    for y in 0..h {
        marg[y + 1] = marg[y] + row_int[y];
    }
    let total = marg[h];
    if total > 0.0 {
        let inv = 1.0 / total;
        for y in 1..=h {
            marg[y] *= inv;
        }
    }
    (marg, cond, total)
}
