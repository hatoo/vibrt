//! Host-side material upload for Principled BSDF.

use crate::gpu_types::{PrincipledGpu, VolumeGpu};
use crate::scene_format::{ColorGraph, ColorNode, PrincipledMaterial, VolumeParams};
use crate::scene_loader::LoadedTexture;
use crate::CudaResultExt;
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::sync::Arc;

fn dptr(slice: &CudaSlice<f32>, stream: &CudaStream) -> optix_sys::CUdeviceptr {
    let (ptr, _sync) = slice.device_ptr(stream);
    ptr as optix_sys::CUdeviceptr
}

fn dptr_u32(slice: &CudaSlice<u32>, stream: &CudaStream) -> optix_sys::CUdeviceptr {
    let (ptr, _sync) = slice.device_ptr(stream);
    ptr as optix_sys::CUdeviceptr
}

/// Upload all textures once; returns parallel vector of device slices + dims.
pub fn upload_textures(
    textures: &[LoadedTexture],
    stream: &Arc<CudaStream>,
    bufs: &mut Vec<CudaSlice<f32>>,
) -> Result<Vec<(optix_sys::CUdeviceptr, i32, i32)>> {
    let mut out = Vec::with_capacity(textures.len());
    for tex in textures {
        let slice = stream.clone_htod(&tex.data[..]).cuda()?;
        let ptr = dptr(&slice, stream);
        out.push((ptr, tex.width as i32, tex.height as i32));
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

pub fn upload_shader_graph(
    graph: &vibrt_nodes::shader::ShaderGraph,
    textures: &[(u64, i32, i32)],
    stream: &Arc<CudaStream>,
    buffers: &mut Vec<CudaSlice<u32>>,
    tables: &mut Vec<CudaSlice<f32>>,
) -> Result<(u64, i32, u32)> {
    graph.validate()?;
    let mut pointers = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        if node.table.is_empty() {
            pointers.push(0);
        } else {
            let data = stream.clone_htod(&node.table).cuda()?;
            pointers.push(dptr(&data, stream));
            tables.push(data);
        }
    }
    let packed = graph.pack(textures, &pointers)?;
    // repr(C), 104 bytes, all fields are integer/f32 with no uninitialized padding.
    let words =
        unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u32>(), packed.len() * 26) };
    let data = stream.clone_htod(words).cuda()?;
    let ptr = dptr_u32(&data, stream);
    buffers.push(data);
    let mut mask = 0;
    for name in graph.outputs.keys() {
        mask |= 1
            << vibrt_nodes::shader::OUTPUTS
                .iter()
                .position(|n| n == name)
                .unwrap();
    }
    Ok((ptr, packed.len() as i32, mask))
}

/// Flatten a `ColorGraph` into the on-device `ColorGraphNode` layout and
/// upload it. Returns a descriptor the material upload can splice in.
pub fn upload_color_graph(
    graph: &ColorGraph,
    textures: &[(optix_sys::CUdeviceptr, i32, i32)],
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
    textures: &[(optix_sys::CUdeviceptr, i32, i32)],
    graph: ColorGraphGpu,
    volume_ptr: optix_sys::CUdeviceptr,
    secondary_ptr: optix_sys::CUdeviceptr,
    left_subtree_ptr: optix_sys::CUdeviceptr,
) -> PrincipledGpu {
    let lookup = |id: Option<u32>| -> (optix_sys::CUdeviceptr, i32, i32, i32) {
        match id {
            Some(i) => match textures.get(i as usize).copied() {
                Some(t) => (t.0, t.1, t.2, 4),
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
        base_color: mat.base_color,
        metallic: mat.metallic.clamp(0.0, 1.0),
        roughness: mat.roughness.clamp(0.0, 1.0),
        ior: mat.ior.max(1.0),
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
        volume: volume_ptr,
        volume_only: if mat.volume_only { 1 } else { 0 },
        pure_glossy: if mat.pure_glossy { 1 } else { 0 },
        pure_diffuse: if mat.pure_diffuse { 1 } else { 0 },
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
