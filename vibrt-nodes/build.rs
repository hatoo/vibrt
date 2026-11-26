const CYCLES: &str = "third_party/blender/intern/cycles";

fn cycles(path: &str) -> String {
    let path = format!("{CYCLES}/{path}");
    println!("cargo:rerun-if-changed={path}");
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("Cannot read {path}: {e}. Run git submodule update --init --depth 1")
    })
}

fn append(source: &mut String, text: &str) {
    for line in text.lines() {
        if !line.trim_start().starts_with("#include")
            && !line.trim_start().starts_with("#  include")
            && !line.starts_with("CCL_NAMESPACE_")
        {
            source.push_str(line);
            source.push('\n');
        }
    }
}

fn main() {
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let mut source =
        String::from("#ifndef VIBRT_CYCLES_NODES_H\n#define VIBRT_CYCLES_NODES_H\n#include \"cycles_compat.h\"\nnamespace vibrt_cycles {\n");
    let types = cycles("kernel/svm/types.h");
    for block in types.split("enum ").skip(1) {
        let end = block.find("};").unwrap();
        if !block[..end].contains("#include") {
            source.push_str("enum ");
            source.push_str(&block[..end + 2]);
            source.push('\n');
        }
    }
    append(
        &mut source,
        &types[types.find("#define CLOSURE_IS_BSDF(type)").unwrap()..],
    );
    let color = cycles("util/color.h");
    append(
        &mut source,
        &color[color.find("ccl_device float3 rgb_to_hsv").unwrap()
            ..color.find("ccl_device float3 xyY_to_xyz").unwrap()],
    );
    for (file, end) in [
        ("tables.h", ""),
        ("hash.h", ""),
        ("volume_util.h", ""),
        ("math_util.h", ""),
        ("color_util.h", ""),
        ("noise.h", ""),
        ("fractal_noise.h", ""),
        ("noisetex.h", "ccl_device_noinline int svm_node_tex_noise"),
        ("voronoi.h", "ccl_device void svm_voronoi_output"),
        ("wave.h", "ccl_device_noinline int svm_node_tex_wave"),
        ("brick.h", "ccl_device_noinline int svm_node_tex_brick"),
        ("checker.h", "ccl_device_noinline void svm_node_tex_checker"),
        (
            "gradient.h",
            "ccl_device_noinline void svm_node_tex_gradient",
        ),
        ("magic.h", "ccl_device_noinline int svm_node_tex_magic"),
        ("gabor.h", "ccl_device_noinline int svm_node_tex_gabor"),
        ("radial_tiling_shared.h", ""),
    ] {
        let mut text = cycles(&format!(
            "{}/{}",
            if file == "hash.h" {
                "util"
            } else if file == "tables.h" {
                "kernel"
            } else if file == "volume_util.h" {
                "kernel/closure"
            } else {
                "kernel/svm"
            },
            file
        ));
        text = text.replace(
            "static inline uint hash_string",
            "ccl_device_inline uint hash_string",
        );
        if !end.is_empty() {
            let offset = text
                .find(end)
                .unwrap_or_else(|| panic!("Missing boundary {end}"));
            text.truncate(offset);
        }
        source.push_str(&format!("\n// BEGIN Cycles v5.1.1 {file}\n"));
        if file == "radial_tiling_shared.h" {
            source.push_str("#define ADAPT_TO_SVM\n");
        }
        for line in text.lines() {
            if !line.starts_with("#include") && !line.starts_with("CCL_NAMESPACE_") {
                source.push_str(line);
                source.push('\n');
            }
        }
        if file == "radial_tiling_shared.h" {
            source.push_str("#undef ADAPT_TO_SVM\n");
        }
    }
    source.push_str(&std::fs::read_to_string("native/closures_compat.h").unwrap());
    let fast_math = cycles("util/math_fast.h");
    let start = fast_math
        .find("ccl_device_inline float fast_ierff")
        .unwrap();
    let end = start + fast_math[start..].find("\n}").unwrap() + 2;
    append(&mut source, &fast_math[start..end]);
    // Numerical closure sources stay unmodified in the pinned submodule.
    source.push_str("\n#ifdef __CUDACC__\n#define VN_TABLE static __device__ const\n#else\n#define VN_TABLE static const\n#endif\n");
    append(
        &mut source,
        &cycles("scene/shader.tables").replace("static const float", "VN_TABLE float"),
    );
    source.push_str("\n#undef VN_TABLE\n");
    source.push_str(&std::fs::read_to_string("native/closure_tables.h").unwrap());
    append(&mut source, &cycles("kernel/util/lookup_table.h"));
    append(&mut source, &cycles("kernel/closure/bsdf_util.h"));
    let mapping = cycles("kernel/sample/mapping.h");
    append(
        &mut source,
        &mapping[..mapping
            .find("ccl_device_inline float pdf_uniform_cone")
            .unwrap()],
    );
    for file in [
        "bsdf_diffuse.h",
        "bsdf_oren_nayar.h",
        "bsdf_sheen.h",
        "bsdf_ashikhmin_velvet.h",
        "bsdf_microfacet.h",
    ] {
        append(&mut source, &cycles(&format!("kernel/closure/{file}")));
    }
    for (file, end) in [(
        "bsdf_principled_hair_chiang.h",
        "/* Implements Filter Glossy",
    )] {
        let text = cycles(&format!("kernel/closure/{file}"));
        for line in text[..text.find(end).unwrap()].lines() {
            if !line.trim_start().starts_with("#  include")
                && !line.starts_with("#include")
                && !line.starts_with("CCL_NAMESPACE_")
            {
                source.push_str(line);
                source.push('\n');
            }
        }
    }
    source.push_str("\n#undef __HAIR__\n#undef SHADER_CLOSURE_BASE\n} // namespace vibrt_cycles\n");
    let bssrdf = cycles("kernel/closure/bssrdf.h");
    let start = bssrdf.find("ccl_device float bssrdf_burley_eval").unwrap();
    let end = bssrdf.find("/* Setup */").unwrap();
    source.push_str("\nnamespace vibrt_cycles {\n#define BURLEY_TRUNCATE 16.0f\n#define BURLEY_TRUNCATE_CDF 0.9963790093708328f\n");
    source.push_str(&bssrdf[start..end]);
    let walk = cycles("kernel/integrator/subsurface_random_walk.h");
    let start = walk
        .find("ccl_device void subsurface_random_walk_remap")
        .unwrap();
    let end = walk
        .find("ccl_device void subsurface_random_walk_coefficients")
        .unwrap();
    source.push_str(&walk[start..end]);
    let start = bssrdf
        .find("ccl_device float bssrdf_dipole_compute_Rd")
        .unwrap();
    let end = bssrdf.find("ccl_device void bssrdf_setup_radius").unwrap();
    source.push_str(&bssrdf[start..end]);
    source.push_str("\n#undef FOREACH_SPECTRUM_CHANNEL\n#undef GET_SPECTRUM_CHANNEL\n}\n");
    source.push_str("\n#endif // VIBRT_CYCLES_NODES_H\n");
    std::fs::write(out.join("cycles_nodes.h"), source).unwrap();
    println!("cargo:rerun-if-changed=native");
    println!("cargo:rerun-if-changed=third_party/sky");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .flag_if_supported("/utf-8")
        .flag_if_supported("-fwrapv")
        .opt_level(2)
        .include("third_party/sky")
        .include("native")
        .include(&out)
        .file("third_party/sky/sky_hosek.cpp")
        .file("third_party/sky/sky_single_scattering.cpp")
        .file("third_party/sky/sky_multiple_scattering.cpp")
        .file("native/sky_bridge.cpp")
        .file("native/color_bridge.cpp")
        .file("native/shader_bridge.cpp")
        .compile("vibrt_nodes");
}
