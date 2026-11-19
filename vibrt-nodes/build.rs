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
    source.push_str("\n} // namespace vibrt_cycles\n");
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
