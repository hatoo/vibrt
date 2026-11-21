fn main() {
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let mut source =
        String::from("#pragma once\n#include \"cycles_compat.h\"\nnamespace vibrt_cycles {\n");
    let types = std::fs::read_to_string("third_party/cycles/types.h").unwrap();
    for block in types.split("enum ").skip(1) {
        let end = block.find("};").unwrap();
        if !block[..end].contains("#include") {
            source.push_str("enum ");
            source.push_str(&block[..end + 2]);
            source.push('\n');
        }
    }
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
        let mut text = std::fs::read_to_string(format!("third_party/cycles/{file}")).unwrap();
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
    for (file, end) in [(
        "bsdf_principled_hair_chiang.h",
        "/* Implements Filter Glossy",
    )] {
        let text = std::fs::read_to_string(format!("third_party/cycles/{file}")).unwrap();
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
    let bssrdf = std::fs::read_to_string("third_party/cycles/bssrdf.h").unwrap();
    let start = bssrdf.find("ccl_device float bssrdf_burley_eval").unwrap();
    let end = bssrdf.find("/* Setup */").unwrap();
    source.push_str("\nnamespace vibrt_cycles {\n#define BURLEY_TRUNCATE 16.0f\n#define BURLEY_TRUNCATE_CDF 0.9963790093708328f\n#define FOREACH_SPECTRUM_CHANNEL(i) for(int i=0;i<3;i++)\n#define GET_SPECTRUM_CHANNEL(v,i) ((v)[i])\n");
    source.push_str(&bssrdf[start..end]);
    let walk = std::fs::read_to_string("third_party/cycles/subsurface_random_walk.h").unwrap();
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
    std::fs::write(out.join("cycles_nodes.h"), source).unwrap();
    println!("cargo:rerun-if-changed=native");
    println!("cargo:rerun-if-changed=third_party/cycles");
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
