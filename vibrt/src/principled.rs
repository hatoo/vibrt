//! Host-side texture, shader-graph, closure-tree, and volume uploads.

use crate::gpu_types::{PrincipledGpu, TextureBinding, VolumeGpu};
use crate::scene_format::{ColorGraph, ColorNode, PrincipledMaterial, VolumeParams};
use crate::scene_loader::{texture_pixel_flags, LoadedTexture};
use crate::CudaResultExt;
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

fn dptr(slice: &CudaSlice<f32>, stream: &CudaStream) -> optix_sys::CUdeviceptr {
    let (ptr, _sync) = slice.device_ptr(stream);
    ptr as optix_sys::CUdeviceptr
}

fn dptr_u32(slice: &CudaSlice<u32>, stream: &CudaStream) -> optix_sys::CUdeviceptr {
    let (ptr, _sync) = slice.device_ptr(stream);
    ptr as optix_sys::CUdeviceptr
}

/// An opaque Mix/Add tree stays opaque regardless of its spatial mix factors.
/// A missing child refers to the current node's own BSDF, as on the device.
fn shader_has_alpha(mat: &PrincipledMaterial) -> bool {
    let leaf_alpha = || {
        mat.alpha_blend < 1.0
            || mat
                .shader_graph
                .as_ref()
                .is_some_and(|g| g.outputs.contains_key("alpha"))
    };
    if mat.left_subtree.is_none() && mat.secondary.is_none() {
        return leaf_alpha();
    }
    mat.left_subtree
        .as_deref()
        .map_or_else(leaf_alpha, shader_has_alpha)
        || mat
            .secondary
            .as_deref()
            .map_or_else(leaf_alpha, shader_has_alpha)
}

/// Only transparent/cutout/volume surfaces need the intersection filter.
/// Be conservative for dynamic transmission and nested material trees.
pub fn needs_any_hit(mat: &PrincipledMaterial) -> bool {
    shader_has_alpha(mat)
        || mat.transmission > 0.0
        || mat.transmission_tex.is_some()
        || mat
            .shader_graph
            .as_ref()
            .is_some_and(|g| g.outputs.contains_key("transmission"))
        || mat.volume.is_some()
        || (mat.alpha_threshold > 0.0 && mat.base_color_tex.is_some())
        || [&mat.left_subtree, &mat.secondary]
            .into_iter()
            .flatten()
            .any(|child| needs_any_hit(child))
}

/// Surface attributes used anywhere in a Mix/Add tree, including its factor.
/// The legacy color evaluator also accepts vertex color and object randomness.
pub fn surface_requirements(mat: &PrincipledMaterial) -> Result<(u32, bool)> {
    let mut contexts = if mat.closure_type == 5 { 1 << 7 } else { 0 };
    let mut bump = false;
    for graph in mat
        .shader_graph
        .iter()
        .chain(mat.aov_graphs.iter().map(|a| &a.graph))
    {
        contexts |= graph.required_contexts()?;
        bump |= graph.outputs.contains_key("bump_height")
            || graph
                .nodes
                .iter()
                .any(|n| matches!(n.op.as_str(), "bump" | "vector_bump" | "shifted_context"));
    }
    if mat.use_vertex_color || mat.color_graph.is_some() {
        contexts |= 1 << 8;
    }
    if mat.color_graph.is_some() {
        contexts |= 1 << 9;
    }
    for child in [&mat.left_subtree, &mat.secondary].into_iter().flatten() {
        let (child_contexts, child_bump) = surface_requirements(child)?;
        contexts |= child_contexts;
        bump |= child_bump;
    }
    Ok((contexts, bump))
}

/// Remove only bit-identical channels; keep the original f32 samples and size.
/// Layouts are luminance, luminance/alpha, RGB and RGBA (1..=4 channels).
/// Reuse initialized storage across uploads; RGBA borrows the source directly.
fn compact_texture<'a>(
    rgba: &'a [f32],
    channel_flags: Option<u32>,
    scratch: &'a mut Vec<f32>,
) -> (&'a [f32], i32) {
    let flags = channel_flags.unwrap_or_else(|| {
        rgba.par_chunks(16_384)
            .map(|chunk| {
                chunk
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .fold(0, |flags, p| flags | texture_pixel_flags(p))
            })
            .reduce(|| 0, |a, b| a | b)
    });
    let channels = match flags {
        0 => 1,
        1 => 3,
        2 => 2,
        _ => return (rgba, 4),
    };
    let len = rgba.len() / 4 * channels;
    if scratch.len() < len {
        scratch.resize(len, 0.0);
    }
    let pixels = &mut scratch[..len];
    pixels
        .par_chunks_mut(4096 * channels)
        .zip(rgba.par_chunks(4096 * 4))
        .for_each(|(dst, src)| {
            for (dst, src) in dst.chunks_exact_mut(channels).zip(src.as_chunks::<4>().0) {
                if channels == 2 {
                    dst.copy_from_slice(&[src[0], src[3]]);
                } else {
                    dst.copy_from_slice(&src[..channels]);
                }
            }
        });
    (pixels, channels as i32)
}

/// Upload all textures once; retain allocations until the render completes.
pub fn upload_textures(
    textures: &[LoadedTexture],
    stream: &Arc<CudaStream>,
    bufs: &mut Vec<CudaSlice<f32>>,
) -> Result<Vec<TextureBinding>> {
    let mut out = Vec::with_capacity(textures.len());
    let mut scratch = Vec::new();
    for tex in textures {
        let (pixels, channels) = compact_texture(tex.data(), tex.channel_flags(), &mut scratch);
        let slice = stream.clone_htod(pixels).cuda()?;
        // The driver may return before consuming the packed pixels. Finish the
        // copy before scratch can be reused, resized or dropped. RGBA continues
        // to borrow the scene's original storage for the whole render.
        if channels != 4 {
            stream.synchronize().cuda()?;
        }
        let ptr = dptr(&slice, stream);
        out.push((ptr, tex.width as i32, tex.height as i32, channels));
        bufs.push(slice);
    }
    Ok(out)
}

/// Info returned by `upload_color_graph` so the caller can stitch it into
/// `PrincipledGpu` (nodes pointer, length, output index).
#[derive(Copy, Clone, Default)]
pub struct ColorGraphGpu {
    pub nodes: optix_sys::CUdeviceptr,
    pub len: i32,
    pub output: i32,
}

#[derive(Clone, Copy)]
pub struct ShaderGraphGpu {
    pub nodes: u64,
    pub len: i32,
    pub outputs: u64,
    pub contexts: u32,
}

/// Render-owned graph buffers. Shared bindings cannot outlive their allocations.
#[derive(Default)]
pub struct GraphStorage {
    pub nodes: Vec<CudaSlice<u32>>,
    pub tables: Vec<CudaSlice<f32>>,
    shaders: HashMap<Vec<u8>, ShaderGraphGpu>,
}

pub fn upload_shader_graph(
    graph: &vibrt_nodes::shader::ShaderGraph,
    textures: &[TextureBinding],
    stream: &Arc<CudaStream>,
    storage: &mut GraphStorage,
) -> Result<ShaderGraphGpu> {
    graph.validate()?;
    // A cache belongs to one render and therefore one set of texture bindings.
    // Exact serialized keys include outputs, image IDs, modes, parameters and LUTs.
    let key = serde_json::to_vec(graph)?;
    if let Some(&binding) = storage.shaders.get(&key) {
        return Ok(binding);
    }
    let mut pointers = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        if node.table.is_empty() {
            pointers.push(0);
        } else {
            let data = stream.clone_htod(&node.table).cuda()?;
            pointers.push(dptr(&data, stream));
            storage.tables.push(data);
        }
    }
    let packed = graph.pack(textures, &pointers)?;
    // repr(C), 104 bytes, all fields are integer/f32 with no uninitialized padding.
    let words =
        unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u32>(), packed.len() * 26) };
    let data = stream.clone_htod(words).cuda()?;
    let ptr = dptr_u32(&data, stream);
    storage.nodes.push(data);
    let mut mask = 0;
    for name in graph.outputs.keys() {
        mask |= 1
            << vibrt_nodes::shader::OUTPUTS
                .iter()
                .position(|n| n == name)
                .unwrap();
    }
    let binding = ShaderGraphGpu {
        nodes: ptr,
        len: packed.len() as i32,
        outputs: mask,
        contexts: graph.required_contexts()?,
    };
    storage.shaders.insert(key, binding);
    Ok(binding)
}

/// Flatten a `ColorGraph` into the on-device `ColorGraphNode` layout and
/// upload it. Returns a descriptor the material upload can splice in.
pub fn upload_color_graph(
    graph: &ColorGraph,
    textures: &[TextureBinding],
    stream: &Arc<CudaStream>,
    bufs: &mut Vec<CudaSlice<u32>>,
    f_bufs: &mut Vec<CudaSlice<f32>>,
) -> Result<ColorGraphGpu> {
    // Constant subgraphs were folded to `const` nodes by the Python exporter
    // (`vibrt_blender/color_fold.py`) before serialisation. The graph
    // arriving here already has them collapsed.
    graph.validate()?;
    // RGBCurve / ColorRamp nodes carry their LUT inline in JSON; allocate
    // them per-node on the GPU so `flatten_color_graph` can splice the
    // device pointer into the payload. RGBCurve has a fixed 768-entry LUT
    // (256 × 3 channels); ColorRamp's stop count is variable and stored
    // alongside the pointer.
    let mut lut_ptrs: Vec<optix_sys::CUdeviceptr> = Vec::with_capacity(graph.nodes.len());
    let mut lut_lens: Vec<u32> = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        match node {
            ColorNode::RgbCurve { lut, .. } => {
                if lut.len() != 768 {
                    return Err(anyhow!(
                        "color_graph RgbCurve.lut has {} entries (want 768)",
                        lut.len()
                    ));
                }
                let slice = stream.clone_htod(lut).cuda()?;
                let ptr = {
                    let (p, _sync) = slice.device_ptr(stream);
                    p as optix_sys::CUdeviceptr
                };
                f_bufs.push(slice);
                lut_ptrs.push(ptr);
                lut_lens.push(0);
            }
            ColorNode::ColorRamp { lut, .. } => {
                if lut.is_empty() || lut.len() % 3 != 0 {
                    return Err(anyhow!(
                        "color_graph ColorRamp.lut has {} floats (need >=3 and multiple of 3)",
                        lut.len()
                    ));
                }
                let slice = stream.clone_htod(lut).cuda()?;
                let ptr = {
                    let (p, _sync) = slice.device_ptr(stream);
                    p as optix_sys::CUdeviceptr
                };
                f_bufs.push(slice);
                lut_ptrs.push(ptr);
                lut_lens.push((lut.len() / 3) as u32);
            }
            _ => {
                lut_ptrs.push(0);
                lut_lens.push(0);
            }
        }
    }
    let flat = vibrt_nodes::graph::pack(graph, textures, &lut_ptrs, &lut_lens)?;
    // ColorGraphNode is repr(C) + 16 × u32 with no padding (see the
    // compile-time size/align asserts in `gpu_types.rs`), so reinterpreting
    // the buffer as `&[u32]` is sound and lets us reuse the existing
    // CudaSlice<u32> storage in `bufs`.
    let flat_u32: &[u32] =
        unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u32, flat.len() * 16) };
    let slice = stream.clone_htod(flat_u32).cuda()?;
    let ptr = dptr_u32(&slice, stream);
    bufs.push(slice);
    let len = graph.nodes.len() as i32;
    let output = match graph.output {
        Some(o) => o as i32,
        None => len - 1,
    };
    if output < 0 || output >= len {
        return Err(anyhow!(
            "color_graph.output = {} out of range [0, {})",
            output,
            len
        ));
    }
    Ok(ColorGraphGpu {
        nodes: ptr,
        len,
        output,
    })
}

/// Precompute the GPU-side volume coefficients from authored params.
/// Cycles' principled volume convention:
///   σ_s = color × density   (RGB scattering coefficient)
///   σ_a = absorption_color × density   (RGB absorption)
///   σ_t = σ_s + σ_a
///   σ_e = emission_color × emission_strength × density
/// Negative or NaN inputs collapse to 0; the device code expects
/// non-negative coefficients.
pub fn make_volume_gpu(v: &VolumeParams) -> VolumeGpu {
    let d = v.density.max(0.0);
    let mut sigma_s = [0.0f32; 3];
    let mut sigma_a = [0.0f32; 3];
    let mut emission = [0.0f32; 3];
    for i in 0..3 {
        sigma_s[i] = (v.color[i].max(0.0)) * d;
        sigma_a[i] = (v.absorption_color[i].max(0.0)) * d;
        emission[i] = (v.emission_color[i].max(0.0)) * v.emission_strength.max(0.0) * d;
    }
    let sigma_t = [
        sigma_s[0] + sigma_a[0],
        sigma_s[1] + sigma_a[1],
        sigma_s[2] + sigma_a[2],
    ];
    VolumeGpu {
        emission_nodes: 0,
        emission_len: 0,
        emission_contexts: 0,
        phase: v.phase,
        phase_params: v.phase_params,
        sigma_t,
        _pad0: 0.0,
        sigma_s,
        _pad1: 0.0,
        emission,
        anisotropy: v.anisotropy.clamp(-0.999, 0.999),
    }
}

/// Build a GPU material struct from a host Principled material.
///
/// `secondary_ptr` and `left_subtree_ptr` are device pointers to the
/// right and left child PrincipledGpu of a binary Mix node (or 0 each
/// for the common single-closure leaf case). When `left_subtree_ptr ==
/// 0` and `secondary_ptr != 0`, the left side defaults to "this node's
/// own lobe params as a leaf" — the right-leaning compact case. When
/// both are non-null this is an internal Mix node and own params are
/// unused. `mix_fac_tex_slot` matches the regular texture-slot tuple
/// — `(ptr, w, h, channels)` — so the caller can pre-resolve the
/// optional mask texture without paying the scene_format / textures
/// slice plumbing inside this helper.
pub fn make_material_data(
    mat: &PrincipledMaterial,
    textures: &[TextureBinding],
    graph: ColorGraphGpu,
    volume_ptr: optix_sys::CUdeviceptr,
    secondary_ptr: optix_sys::CUdeviceptr,
    left_subtree_ptr: optix_sys::CUdeviceptr,
) -> PrincipledGpu {
    let lookup = |id: Option<u32>| -> (optix_sys::CUdeviceptr, i32, i32, i32) {
        match id {
            Some(i) => match textures.get(i as usize).copied() {
                Some(t) => t,
                None => {
                    eprintln!(
                        "[vibrt] warn: material texture index {} out of range \
                         (have {} textures) — using null texture",
                        i,
                        textures.len()
                    );
                    (0, 0, 0, 0)
                }
            },
            None => (0, 0, 0, 0),
        }
    };
    let (bc_ptr, bc_w, bc_h, bc_c) = lookup(mat.base_color_tex);
    let (n_ptr, n_w, n_h, n_c) = lookup(mat.normal_tex);
    let (r_ptr, r_w, r_h, r_c) = lookup(mat.roughness_tex);
    let (m_ptr, m_w, m_h, m_c) = lookup(mat.metallic_tex);
    let (t_ptr, t_w, t_h, t_c) = lookup(mat.transmission_tex);
    let (e_ptr, e_w, e_h, e_c) = lookup(mat.emission_tex);
    let (b_ptr, b_w, b_h, b_c) = lookup(mat.bump_tex);
    let (mf_ptr, mf_w, mf_h, mf_c) = lookup(mat.mix_fac_tex);

    PrincipledGpu {
        surface_model: mat.surface_model,
        distribution: mat.distribution,
        specular_tint: mat.specular_tint,
        coat_tint: mat.coat_tint,
        coat_normal: mat.coat_normal,
        thin_film_thickness: mat.thin_film_thickness,
        thin_film_ior: mat.thin_film_ior,
        conductor_ior: mat.conductor_ior,
        conductor_k: mat.conductor_k,
        diffuse_roughness: mat.diffuse_roughness.clamp(0.0, 1.0),
        aov_shaders: 0,
        num_aov_shaders: 0,
        base_color: mat.base_color,
        metallic: mat.metallic.clamp(0.0, 1.0),
        roughness: mat.roughness.clamp(0.0, 1.0),
        ior: mat.ior.max(1e-5),
        transmission: mat.transmission.clamp(0.0, 1.0),
        specular_ior_level: mat.specular_ior_level.max(0.0),
        emission: mat.emission,
        base_color_tex: bc_ptr,
        base_color_tex_w: bc_w,
        base_color_tex_h: bc_h,
        base_color_tex_channels: bc_c,
        normal_tex: n_ptr,
        normal_tex_w: n_w,
        normal_tex_h: n_h,
        normal_tex_channels: n_c,
        roughness_tex: r_ptr,
        roughness_tex_w: r_w,
        roughness_tex_h: r_h,
        roughness_tex_channels: r_c,
        metallic_tex: m_ptr,
        metallic_tex_w: m_w,
        metallic_tex_h: m_h,
        metallic_tex_channels: m_c,
        transmission_tex: t_ptr,
        transmission_tex_w: t_w,
        transmission_tex_h: t_h,
        transmission_tex_channels: t_c,
        emission_tex: e_ptr,
        emission_tex_w: e_w,
        emission_tex_h: e_h,
        emission_tex_channels: e_c,
        uv_transform: mat.uv_transform,
        normal_strength: mat.normal_strength,
        bump_strength: mat.bump_strength,
        bump_tex: b_ptr,
        bump_tex_w: b_w,
        bump_tex_h: b_h,
        bump_tex_channels: b_c,
        alpha_threshold: mat.alpha_threshold.clamp(0.0, 1.0),
        anisotropy: mat.anisotropy.clamp(-1.0, 1.0),
        tangent_rotation: mat.tangent_rotation,
        coat_weight: mat.coat_weight.clamp(0.0, 1.0),
        coat_roughness: mat.coat_roughness.clamp(0.0, 1.0),
        coat_ior: mat.coat_ior.max(1.0),
        sheen_weight: mat.sheen_weight.clamp(0.0, 1.0),
        sheen_roughness: mat.sheen_roughness.clamp(0.0, 1.0),
        sheen_tint: mat.sheen_tint,
        sss_weight: mat.sss_weight.clamp(0.0, 1.0),
        sss_radius: mat.sss_radius,
        sss_method: mat.sss_method,
        sss_scale: mat.sss_scale.max(0.0),
        sss_ior: mat.sss_ior.clamp(1.01, 3.8),
        sss_anisotropy: mat.sss_anisotropy.clamp(0.0, 0.9),
        translucent_weight: mat.translucent_weight.clamp(0.0, 1.0),
        hair_weight: mat.hair_weight.clamp(0.0, 1.0),
        hair_offset: mat.hair_offset,
        hair_roughness_u: mat.hair_roughness_u.clamp(1e-3, 1.0),
        hair_roughness_v: mat.hair_roughness_v.clamp(1e-3, 1.0),
        use_vertex_color: if mat.use_vertex_color { 1 } else { 0 },
        color_graph_nodes: graph.nodes,
        color_graph_len: graph.len,
        color_graph_output: graph.output,
        shader_nodes: 0,
        shader_len: 0,
        shader_outputs: 0,
        shader_contexts: 0,
        shader_has_alpha: i32::from(shader_has_alpha(mat)),
        surface_contexts: 0,
        surface_bump: 0,
        volume: volume_ptr,
        volume_only: if mat.volume_only { 1 } else { 0 },
        pure_glossy: if mat.pure_glossy { 1 } else { 0 },
        pure_diffuse: if mat.pure_diffuse { 1 } else { 0 },
        closure_type: mat.closure_type,
        alpha_blend: mat.alpha_blend.clamp(0.0, 1.0),
        secondary: secondary_ptr,
        left_subtree: left_subtree_ptr,
        // A negative mix_fac is the view-dependent Fresnel sentinel (its
        // magnitude is the Fresnel IOR); leave it untouched. Only clamp the
        // ordinary constant-factor case to [0, 1].
        add_shader: mat.add_shader as i32,
        mix_fac: if mat.mix_fac < 0.0 {
            mat.mix_fac
        } else {
            mat.mix_fac.clamp(0.0, 1.0)
        },
        mix_fac_tex: mf_ptr,
        mix_fac_tex_w: mf_w,
        mix_fac_tex_h: mf_h,
        mix_fac_tex_channels: mf_c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn opaque_materials_do_not_need_intersection_filtering() {
        let opaque = json!({"base_color":[0.5,0.5,0.5]});
        let mut mat: PrincipledMaterial = serde_json::from_value(opaque.clone()).unwrap();
        assert!(!needs_any_hit(&mat));
        // A base-color image alone does not enable alpha cutout.
        mat.base_color_tex = Some(0);
        assert!(!needs_any_hit(&mat));
        mat.alpha_threshold = 0.5;
        assert!(needs_any_hit(&mat));
        for transparent in [
            json!({"base_color":[1,1,1],"alpha_blend":0.5}),
            json!({"base_color":[1,1,1],"transmission":1.0}),
            json!({"base_color":[1,1,1],"transmission_tex":0}),
            json!({"base_color":[1,1,1],"volume":{},"volume_only":true}),
            json!({"base_color":[1,1,1],"shader_graph":{
                "nodes":[{"op":"context"}],"outputs":{"alpha":0}}}),
            json!({"base_color":[1,1,1],"shader_graph":{
                "nodes":[{"op":"context"}],"outputs":{"transmission":0}}}),
        ] {
            assert!(needs_any_hit(
                &serde_json::from_value(transparent.clone()).unwrap()
            ));
            for child in ["left_subtree", "secondary"] {
                let mut tree = opaque.clone();
                tree[child] = opaque.clone();
                assert!(!needs_any_hit(
                    &serde_json::from_value(tree.clone()).unwrap()
                ));
                tree[child] = transparent.clone();
                assert!(needs_any_hit(&serde_json::from_value(tree).unwrap()));
            }
        }
    }

    #[test]
    fn texture_compaction_preserves_float_bits_and_alpha() {
        let mut scratch = Vec::new();
        for (rgba, expected_channels) in [
            (vec![0.25, 0.25, 0.25, 1.0], 1),
            (vec![0.25, 0.25, 0.25, 0.4], 2),
            (vec![0.25, 0.5, 8.0, 1.0], 3),
            (vec![0.25, 0.5, 8.0, 0.4], 4),
            (vec![-0.0, 0.0, -0.0, 1.0], 3),
            (vec![f32::INFINITY, 1.0, f32::NAN, 0.0], 4),
        ] {
            let expected = [rgba[0], rgba[1], rgba[2], rgba[3]].map(f32::to_bits);
            // Exercise the boundary between parallel chunks as well as small inputs.
            for repeat in [1, 4097] {
                let source = rgba.repeat(repeat);
                let (packed, channels) = compact_texture(&source, None, &mut scratch);
                assert_eq!(channels, expected_channels);
                assert_eq!(packed.len(), repeat * channels as usize);
                for pixel in packed.chunks_exact(channels as usize) {
                    let restored = match channels {
                        1 => [pixel[0], pixel[0], pixel[0], 1.0],
                        2 => [pixel[0], pixel[0], pixel[0], pixel[1]],
                        3 => [pixel[0], pixel[1], pixel[2], 1.0],
                        _ => [pixel[0], pixel[1], pixel[2], pixel[3]],
                    };
                    assert_eq!(restored.map(f32::to_bits), expected);
                }
            }
        }
        let mut mixed = vec![0.0, 0.0, 0.0, 1.0].repeat(8193);
        mixed[4096 * 4 + 1] = 0.25;
        mixed[8192 * 4 + 3] = 0.5;
        let (packed, channels) = compact_texture(&mixed, None, &mut scratch);
        assert_eq!(channels, 4);
        assert_eq!(packed.as_ptr(), mixed.as_ptr());
    }

    #[test]
    fn texture_preclassified_compaction_matches_scanned_layouts() {
        let mut scanned_scratch = Vec::new();
        let mut cached_scratch = Vec::new();
        for (pixel, flags) in [
            ([0.25, 0.25, 0.25, 1.0], 0),
            ([0.25, 0.5, 8.0, 1.0], 1),
            ([f32::from_bits(0x7fc0_0001); 4], 2),
            ([-0.0, 0.0, -0.0, 0.5], 3),
        ] {
            for count in [1, 4097, 8193] {
                let source = pixel.repeat(count);
                let (scanned, scan_channels) = compact_texture(&source, None, &mut scanned_scratch);
                let (cached, cache_channels) =
                    compact_texture(&source, Some(flags), &mut cached_scratch);
                assert_eq!(scan_channels, cache_channels);
                assert_eq!(
                    scanned.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    cached.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn texture_scratch_reuse_preserves_pixels_across_sizes_and_layouts() {
        // Repeatedly grow and reuse the same storage, including smaller images
        // and a borrowed RGBA image between packed uploads. Vary every pixel so
        // stale channels, stale tails and parallel chunk boundaries are visible.
        for threads in [1, 2, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut scratch = Vec::new();
                    for (round, count) in [8193, 3, 12289, 4097, 0, 1, 8192].into_iter().enumerate()
                    {
                        for layout in [3, 1, 4, 2, 3, 2, 1] {
                            let mut rgba = Vec::with_capacity(count * 4);
                            for i in 0..count {
                                let value = match i % 5 {
                                    0 => -0.0,
                                    1 => f32::INFINITY,
                                    2 => f32::from_bits(0x7fc0_0001 + round as u32),
                                    _ => (i + round) as f32 * 0.001,
                                };
                                let (g, b) = if layout <= 2 {
                                    (value, value)
                                } else {
                                    (value + 0.25, value + 0.5)
                                };
                                let alpha = if layout == 2 || layout == 4 { 0.5 } else { 1.0 };
                                rgba.extend_from_slice(&[value, g, b, alpha]);
                            }
                            let (packed, channels) = compact_texture(&rgba, None, &mut scratch);
                            assert_eq!(channels, if count == 0 { 1 } else { layout });
                            assert_eq!(packed.len(), count * channels as usize);
                            for (pixel, expected) in packed
                                .chunks_exact(channels as usize)
                                .zip(rgba.as_chunks::<4>().0)
                            {
                                let restored = match channels {
                                    1 => [pixel[0], pixel[0], pixel[0], 1.0],
                                    2 => [pixel[0], pixel[0], pixel[0], pixel[1]],
                                    3 => [pixel[0], pixel[1], pixel[2], 1.0],
                                    _ => [pixel[0], pixel[1], pixel[2], pixel[3]],
                                };
                                assert_eq!(restored.map(f32::to_bits), expected.map(f32::to_bits));
                            }
                        }
                    }
                });
        }
    }

    #[test]
    fn surface_inputs_include_factors_nested_leaves_and_legacy_colors() {
        let mat: PrincipledMaterial = serde_json::from_value(json!({
            "base_color":[1,1,1],
            "shader_graph":{"nodes":[{"op":"context","modes":[16,0,0,0]}],
                            "outputs":{"mix_fac":0}},
            "left_subtree":{"base_color":[1,1,1],"use_vertex_color":true},
            "secondary":{"base_color":[1,1,1],
                "shader_graph":{"nodes":[{"op":"context","modes":[1,0,0,0]}],
                                "outputs":{"bump_height":0}},
                "secondary":{"base_color":[1,1,1],
                    "shader_graph":{"nodes":[{"op":"context","modes":[13,0,0,0]}],
                                    "outputs":{"base_color":0}}}}
        }))
        .unwrap();
        assert_eq!(
            surface_requirements(&mat).unwrap(),
            ((1 << 1) | (1 << 8) | (1 << 13) | (1 << 16), true)
        );
        let plain = serde_json::from_value(json!({"base_color":[1,1,1]})).unwrap();
        assert_eq!(surface_requirements(&plain).unwrap(), (0, false));
    }

    #[test]
    fn opaque_trees_ignore_unused_internal_leaf_parameters() {
        for add in [false, true] {
            let mat = serde_json::from_value(json!({
                "base_color":[1,1,1], "alpha_blend":0.0, "add_shader":add,
                "shader_graph":{"nodes":[{"op":"constant","params":[0.3,0,0,0]}],
                                "outputs":{"mix_fac":0,"alpha":0}},
                "left_subtree":{"base_color":[1,1,1]},
                "secondary":{"base_color":[1,1,1],
                    "secondary":{"base_color":[1,1,1]}}
            }))
            .unwrap();
            assert!(!shader_has_alpha(&mat));
        }
    }

    #[test]
    fn opacity_follows_dynamic_leaves_and_missing_child_fallbacks() {
        for child in ["left_subtree", "secondary"] {
            let mut value = json!({"base_color":[1,1,1],"alpha_blend":0.25});
            value[child] = json!({"base_color":[1,1,1]});
            assert!(shader_has_alpha(&serde_json::from_value(value).unwrap()));
        }
        let mut value = json!({"base_color":[1,1,1],
            "left_subtree":{"base_color":[1,1,1]},
            "secondary":{"base_color":[1,1,1]}});
        value["secondary"]["shader_graph"] = json!({
            "nodes":[{"op":"context","modes":[8,0,0,0]}],"outputs":{"alpha":0}});
        assert!(shader_has_alpha(
            &serde_json::from_value(value.clone()).unwrap()
        ));
        value["secondary"]["shader_graph"]["outputs"] = json!({"base_color":0});
        assert!(!shader_has_alpha(
            &serde_json::from_value(value.clone()).unwrap()
        ));
        value["secondary"]["alpha_blend"] = json!(0.5);
        assert!(shader_has_alpha(&serde_json::from_value(value).unwrap()));
    }
}
