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

// Keep selected upstream functions byte-for-byte, with a checked boundary.
fn function(source: &mut String, path: &str, signature: &str) {
    let text = cycles(path);
    let start = text
        .find(signature)
        .unwrap_or_else(|| panic!("Missing {signature} in {path}"));
    let body = start + text[start..].find('{').unwrap();
    let mut depth = 0;
    for (offset, byte) in text[body..].bytes().enumerate() {
        if byte == b'{' {
            depth += 1;
        }
        if byte == b'}' {
            depth -= 1;
        }
        if depth == 0 {
            append(source, &text[start..body + offset + 1]);
            return;
        }
    }
    panic!("Unclosed {signature} in {path}");
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
    source.push_str(&std::fs::read_to_string("native/reuse_compat.h").unwrap());
    function(
        &mut source,
        "util/math_base.h",
        "ccl_device_inline float one_minus_cos(",
    );
    function(
        &mut source,
        "util/math_base.h",
        "ccl_device_inline float cubic_interp(",
    );
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
    append(&mut source, &cycles("kernel/sample/mis.h"));
    for signature in [
        "ccl_device float svm_bevel_cubic_eval(",
        "ccl_device float svm_bevel_cubic_pdf(",
        "ccl_device_forceinline float svm_bevel_cubic_quintic_root_find(",
        "ccl_device void svm_bevel_cubic_sample(",
    ] {
        function(&mut source, "kernel/svm/bevel.h", signature);
    }
    let mapping = cycles("kernel/sample/mapping.h");
    append(
        &mut source,
        &mapping[..mapping
            .find("ccl_device_inline float pdf_uniform_cone")
            .unwrap()],
    );
    function(
        &mut source,
        "kernel/sample/mapping.h",
        "ccl_device_inline float3 sample_uniform_cone",
    );
    function(
        &mut source,
        "kernel/sample/mapping.h",
        "ccl_device float2 regular_polygon_sample(",
    );
    source
        .push_str("\nstruct KernelCamera { float blades, bladesrotation, inv_aperture_ratio; };\n");
    function(
        &mut source,
        "kernel/camera/camera.h",
        "ccl_device float2 camera_sample_aperture(",
    );
    for file in [
        "bsdf_diffuse.h",
        "bsdf_oren_nayar.h",
        "bsdf_sheen.h",
        "bsdf_ashikhmin_velvet.h",
        "bsdf_microfacet.h",
        "bsdf_hair.h",
        "bsdf_toon.h",
    ] {
        append(&mut source, &cycles(&format!("kernel/closure/{file}")));
    }
    for signature in [
        "ccl_device_inline float3 transform_direction(",
        "ccl_device_inline float3 transform_direction_transposed(",
        "ccl_device_inline Transform make_transform(",
        "ccl_device_inline Transform euler_to_transform(",
    ] {
        function(&mut source, "util/transform.h", signature);
    }
    function(
        &mut source,
        "util/math_float3.h",
        "ccl_device_inline float3 rotate_around_axis(",
    );
    for file in ["mapping_util.h", "map_range.h", "vector_rotate.h"] {
        append(&mut source, &cycles(&format!("kernel/svm/{file}")));
    }
    source.push_str(&std::fs::read_to_string("native/hair_adapter.h").unwrap());
    function(
        &mut source,
        "kernel/sample/util.h",
        "ccl_device_inline uint reversed_bit_owen(",
    );
    append(&mut source, &cycles("kernel/sample/sobol_burley.h"));
    function(
        &mut source,
        "util/math_intersect.h",
        "ccl_device_forceinline bool ray_triangle_intersect(",
    );
    function(
        &mut source,
        "util/math_intersect.h",
        "ccl_device_forceinline float2 ray_triangle_uv(",
    );
    function(
        &mut source,
        "kernel/bvh/util.h",
        "ccl_device_inline float3 ray_offset(",
    );
    function(
        &mut source,
        "kernel/closure/volume.h",
        "ccl_device Spectrum volume_color_transmittance(",
    );
    for signature in [
        "ccl_device_forceinline float eval_phase_dwivedi(",
        "ccl_device_forceinline float sample_phase_dwivedi(",
        "ccl_device_forceinline float diffusion_length_dwivedi(",
        "ccl_device_forceinline Spectrum subsurface_random_walk_pdf(",
    ] {
        function(
            &mut source,
            "kernel/integrator/subsurface_random_walk.h",
            signature,
        );
    }
    let math_base = cycles("util/math_base.h");
    source.push_str(
        &math_base[math_base.find("/* Defines a closed interval").unwrap()
            ..math_base
                .find("/* Defines the minimal and maximal")
                .unwrap()],
    );
    for (file, end) in [
        (
            "bsdf_principled_hair_chiang.h",
            "/* Implements Filter Glossy",
        ),
        (
            "bsdf_principled_hair_huang.h",
            "/* Implements Filter Glossy",
        ),
    ] {
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
    source.push_str("\nnamespace vibrt_camera {\nusing namespace vibrt_cycles;\nusing KernelGlobals = const float *;\n#undef kernel_data_fetch\n#define kernel_data_fetch(name, index) kg[index]\n");
    function(
        &mut source,
        "kernel/util/lookup_table.h",
        "ccl_device float lookup_table_read(",
    );
    source.push_str("\n#undef kernel_data_fetch\n#define kernel_data_fetch(name, index) closure_table_fetch(index)\n}\n");
    source.push_str("\nnamespace vibrt_image {\n");
    source.push_str(&std::fs::read_to_string("native/image_compat.h").unwrap());
    let image = cycles("kernel/device/cpu/image.h");
    append(
        &mut source,
        &image[image.find("#define SET_CUBIC_SPLINE_WEIGHTS").unwrap()
            ..image.find("template<typename TexT").unwrap()],
    );
    source.push_str("\nstruct Interpolator : ImageReader {\n");
    for signature in [
        "static ccl_always_inline int wrap_periodic(",
        "static ccl_always_inline int wrap_clamp(",
        "static ccl_always_inline int wrap_mirror(",
        "static ccl_always_inline OutT interp_closest(",
        "static ccl_always_inline OutT interp_linear(",
        "static ccl_always_inline OutT interp_cubic(",
        "static ccl_always_inline OutT interp(",
    ] {
        function(&mut source, "kernel/device/cpu/image.h", signature);
    }
    source.push_str("\n};\n#undef SET_CUBIC_SPLINE_WEIGHTS\n}\n");
    source.push_str("\nnamespace vibrt_lights {\n");
    source.push_str(&std::fs::read_to_string("native/lights_compat.h").unwrap());
    for (path, signature) in [
        (
            "util/math_base.h",
            "ccl_device_inline float sin_sqr_to_one_minus_cos(",
        ),
        ("util/math_float3.h", "ccl_device_inline float tan_angle("),
        (
            "util/math_intersect.h",
            "ccl_device bool ray_sphere_intersect(",
        ),
        (
            "kernel/sample/mapping.h",
            "ccl_device float3 sample_uniform_sphere(",
        ),
        (
            "kernel/light/common.h",
            "ccl_device_inline float3 ellipse_sample(",
        ),
        (
            "kernel/light/common.h",
            "ccl_device float3 disk_light_sample(",
        ),
        (
            "kernel/light/common.h",
            "ccl_device float light_pdf_area_to_solid_angle(",
        ),
        (
            "kernel/light/point.h",
            "ccl_device_inline bool point_light_sample(",
        ),
        (
            "kernel/light/point.h",
            "ccl_device_forceinline float sphere_light_pdf(",
        ),
        (
            "kernel/light/spot.h",
            "ccl_device float3 spot_light_to_local(",
        ),
        (
            "kernel/light/spot.h",
            "ccl_device float spot_light_attenuation(",
        ),
    ] {
        function(&mut source, path, signature);
    }
    source.push_str("\ntemplate<bool in_volume_segment>\n");
    function(
        &mut source,
        "kernel/light/spot.h",
        "ccl_device_inline bool spot_light_sample(",
    );
    for (path, signature) in [
        (
            "kernel/light/spot.h",
            "ccl_device_forceinline float spot_light_pdf(",
        ),
        (
            "kernel/light/distant.h",
            "ccl_device_inline bool distant_light_sample(",
        ),
        ("util/math_fast.h", "ccl_device_inline float vector_angle("),
        (
            "kernel/light/distant.h",
            "ccl_device LightEval distant_light_eval_from_intersection(",
        ),
        (
            "kernel/light/area.h",
            "ccl_device_inline float area_light_rect_sample(",
        ),
        (
            "kernel/light/area.h",
            "ccl_device float area_light_spread_attenuation(",
        ),
        (
            "kernel/light/area.h",
            "ccl_device bool area_light_spread_clamp_light(",
        ),
    ] {
        function(&mut source, path, signature);
    }
    source.push_str("\n}\n");
    source.push_str("\nnamespace vibrt_differential {\nusing namespace vibrt_cycles;\nusing float3 = vibrt_cycles::float3;\nstruct differential { float dx, dy; };\nstruct differential3 { float3 dx, dy; };\n");
    for signature in [
        "ccl_device void differential_dudv(",
        "ccl_device_forceinline float differential_transfer_compact(",
        "ccl_device_forceinline differential3 differential_from_compact(",
    ] {
        function(&mut source, "kernel/util/differential.h", signature);
    }
    source.push_str("\n}\nnamespace vibrt_scene_nodes {\n");
    source.push_str(&std::fs::read_to_string("native/scene_nodes_compat.h").unwrap());
    source.push_str("\n#define __RAY_DIFFERENTIALS__\n#define IF_KERNEL_NODES_FEATURE(x) if (true)\ntemplate<uint node_feature_mask>\n");
    function(
        &mut source,
        "kernel/svm/displace.h",
        "ccl_device_noinline int svm_node_set_bump(",
    );
    function(
        &mut source,
        "kernel/svm/tex_coord.h",
        "ccl_device_noinline void svm_node_normal_map(",
    );
    source.push_str("\n#define __HAIR__\n");
    function(
        &mut source,
        "kernel/svm/wireframe.h",
        "ccl_device_inline float wireframe(",
    );
    source.push_str(
        "\n#undef __HAIR__\n#undef IF_KERNEL_NODES_FEATURE\n#undef __RAY_DIFFERENTIALS__\n",
    );
    source.push_str(&std::fs::read_to_string("native/scene_nodes_adapter.h").unwrap());
    source.push_str("\n}\n");
    source.push_str("\nnamespace vibrt_shadow {\nusing namespace vibrt_cycles;\nconstexpr int SD_HAS_ONLY_VOLUME = 1, SD_TRANSPARENT = 2, SD_RAY_PORTAL = 4;\nstruct ShaderData { int flag; Spectrum closure_transparent_extinction; };\n");
    function(
        &mut source,
        "kernel/integrator/surface_shader.h",
        "ccl_device Spectrum surface_shader_transparency(",
    );
    source.push_str("\n}\n");
    source.push_str("\nnamespace vibrt_adaptive {\n");
    source.push_str(&std::fs::read_to_string("native/adaptive_compat.h").unwrap());
    for signature in [
        "ccl_device bool film_adaptive_sampling_convergence_check(",
        "ccl_device void film_adaptive_sampling_filter_x(",
        "ccl_device void film_adaptive_sampling_filter_y(",
    ] {
        function(&mut source, "kernel/film/adaptive_sampling.h", signature);
    }
    function(
        &mut source,
        "kernel/sample/pattern.h",
        "ccl_device_inline bool sample_is_class_A(",
    );
    source.push_str("\n#undef kernel_data\n#define kernel_data ClosureKernelData{}\n}\n");
    source.push_str("\nnamespace vibrt_volume {\nusing namespace vibrt_cycles;\nusing float3 = vibrt_cycles::float3;\nstruct Ray { float3 P, D; };\nstruct EquiangularCoefficients { float3 P; struct { float min, max; } t_range; };\n");
    for signature in [
        "ccl_device float volume_equiangular_sample(",
        "ccl_device float volume_equiangular_pdf(",
    ] {
        function(&mut source, "kernel/integrator/shade_volume.h", signature);
    }
    source.push_str("\n}\n");
    source.push_str("\nnamespace vibrt_background {\n");
    source.push_str(&std::fs::read_to_string("native/background_compat.h").unwrap());
    for signature in [
        "ccl_device float3 background_map_sample(",
        "ccl_device float background_map_pdf(",
    ] {
        function(&mut source, "kernel/light/background.h", signature);
    }
    source.push_str("\n#undef kernel_data\n#define kernel_data ClosureKernelData{}\n#undef kernel_data_fetch\n#define kernel_data_fetch(name, index) closure_table_fetch(index)\n}\n");
    source.push_str("\nnamespace vibrt_ies {\nusing namespace vibrt_cycles;\nusing vibrt_cycles::__float_as_int;\nusing KernelGlobals = const float *;\n#undef kernel_data_fetch\n#define kernel_data_fetch(name, index) kg[index]\n");
    append(&mut source, &cycles("kernel/util/ies.h"));
    source.push_str("\n#undef kernel_data_fetch\n#define kernel_data_fetch(name, index) closure_table_fetch(index)\n}\n");
    source.push_str("\nnamespace vibrt_triangle {\n");
    source.push_str(&std::fs::read_to_string("native/triangle_compat.h").unwrap());
    for signature in [
        "ccl_device_inline float triangle_light_pdf_area_sampling(",
        "ccl_device_forceinline float triangle_light_pdf(",
        "ccl_device_forceinline bool triangle_light_sample(",
    ] {
        if signature.contains("bool triangle_light_sample") {
            source.push_str("\ntemplate<bool in_volume_segment>\n");
        }
        function(&mut source, "kernel/light/triangle.h", signature);
    }
    source.push_str("\n}\nnamespace vibrt_light_tree {\nusing namespace vibrt_cycles;\nusing float3=vibrt_cycles::float3;\nusing vibrt_cycles::make_float3;\nstruct KernelBoundingBox { float3 min, max; };\nstruct KernelBoundingCone { float3 axis; float theta_o, theta_e; };\n");
    function(
        &mut source,
        "kernel/light/tree.h",
        "ccl_device float light_tree_cos_bound_subtended_angle(",
    );
    source.push_str("\ntemplate<bool in_volume_segment>\n");
    function(
        &mut source,
        "kernel/light/tree.h",
        "ccl_device void light_tree_importance(",
    );
    source.push_str(&std::fs::read_to_string("native/light_tree_compat.h").unwrap());
    source.push_str("\ntemplate<bool in_volume_segment>\n");
    function(
        &mut source,
        "kernel/light/tree.h",
        "ccl_device bool get_left_probability(",
    );
    source.push_str(&std::fs::read_to_string("native/light_tree_traversal.h").unwrap());
    source.push_str("\n#undef kernel_data\n#define kernel_data ClosureKernelData{}\n#undef kernel_data_fetch\n#define kernel_data_fetch(name, index) closure_table_fetch(index)\n}\n#endif // VIBRT_CYCLES_NODES_H\n");
    std::fs::write(out.join("cycles_nodes.h"), source).unwrap();
    let mut host = String::from("#include <algorithm>\n#include <array>\n#include <vector>\n#include <cassert>\n#include \"cycles_nodes.h\"\nnamespace vibrt_host {\nusing namespace vibrt_cycles;\nusing std::vector;\nusing std::upper_bound;\n");
    append(&mut host, &cycles("util/math_cdf.h"));
    append(&mut host, &cycles("util/math_cdf.cpp"));
    host.push_str("\nenum FilterType { FILTER_BOX, FILTER_GAUSSIAN, FILTER_BLACKMAN_HARRIS };\nconstexpr int FILTER_TABLE_SIZE = 1024;\n");
    for signature in [
        "static float filter_func_box(",
        "static float filter_func_gaussian(",
        "static float filter_func_blackman_harris(",
        "static vector<float> filter_table(",
    ] {
        function(&mut host, "scene/film.cpp", signature);
    }
    host.push_str(&std::fs::read_to_string("native/light_tree_host_compat.h").unwrap());
    let tree = cycles("scene/light_tree.h");
    append(
        &mut host,
        &tree[tree.find("struct OrientationBounds {").unwrap()
            ..tree
                .find("/* --------------------------------------------------------------------")
                .unwrap()],
    );
    append(
        &mut host,
        &tree[tree.find("struct LightTreeMeasure {").unwrap()
            ..tree.find("  bool transform(").unwrap()],
    );
    host.push_str("\n};\n");
    append(
        &mut host,
        &tree[tree.find("LightTreeMeasure operator+").unwrap()
            ..tree.find("struct LightTreeEmitter {").unwrap()],
    );
    host.push_str("\nstruct LightTreeEmitter { float3 centroid; uint64_t light_set_membership = 1; LightTreeMeasure measure; int index; };\n");
    append(
        &mut host,
        &tree[tree.find("struct LightTreeBucket {").unwrap()
            ..tree.find("/* Light Tree Node */").unwrap()],
    );
    host.push_str("\nstruct LightTree { int max_lights_in_leaf_ = 1; bool should_split(LightTreeEmitter *, int, int &, int, LightTreeMeasure &, LightTreeLightLink &, int &); };\n");
    for signature in [
        "float OrientationBounds::calculate_measure()",
        "OrientationBounds merge(",
        "bool LightTree::should_split(",
        "__forceinline LightTreeMeasure operator+(",
        "LightTreeBucket operator+(",
        "LightTreeLightLink operator+(",
    ] {
        function(&mut host, "scene/light_tree.cpp", signature);
    }
    host.push_str("\n}\n");
    std::fs::write(out.join("cycles_host.h"), host).unwrap();
    let mut ies = String::from("#include <string>\n#include <vector>\n#include <algorithm>\n#include <cstring>\n#include <cstdlib>\n#include <cmath>\n#include <limits>\nnamespace vibrt_ies_host {\nusing std::string;\nusing std::vector;\nconstexpr float M_PI_F = 3.14159265358979323846f;\ninline float __int_as_float(int v) { float f; std::memcpy(&f, &v, 4); return f; }\n");
    append(&mut ies, &cycles("util/ies.h"));
    append(
        &mut ies,
        &cycles("util/ies.cpp").replace("template class GuardedAllocator<char>;", ""),
    );
    ies.push_str("\n}\n");
    std::fs::write(out.join("cycles_ies_host.h"), ies).unwrap();
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
        .file("native/ies_bridge.cpp")
        .file("native/reuse_bridge.cpp")
        .file("native/host_bridge.cpp")
        .compile("vibrt_nodes");
}
