//! GPU scene ownership. Uploads never borrow Blender or Python buffers.
use super::{mesh_uploads, query_cache, session::RenderSession, upload::*};
use crate::{
    camera,
    gpu_types::*,
    material::ShaderRequirements,
    principled, scene_format,
    scene_loader::{self, LoadedScene},
    shader_compile::CompiledShaders,
    CudaResultExt, Progress,
};
use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use optix::{
    accel::{self, AccelBuildOptions, BuildInput, TriangleArrayInput},
    *,
};
use std::{
    mem,
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};

/// Owns a complete uploaded scene and borrows only its GPU session.
/// Multiple RenderJobs can use it without retaining source arrays or reuploading.
pub struct GpuScene<'session> {
    session: &'session RenderSession,
    shaders: Rc<CompiledShaders>,
    pub(super) params: LaunchParams,
    sbt: optix_sys::OptixShaderBindingTable,
    camera: scene_format::CameraDesc,
    pub(super) settings: scene_format::RenderSettings,
    aov_names: Vec<String>,
    pub(super) ptx_time: Duration,
    pub(super) pipeline_time: Duration,
    pub(super) upload_time: Duration,
    _uploads: UploadArena,
    _textures: Vec<CudaSlice<f32>>,
    _graphs: principled::GraphStorage,
    _accels: Vec<CudaSlice<u8>>,
    _meshes: mesh_uploads::MeshUploads,
    _material_tables: Vec<CudaSlice<u64>>,
}

impl GpuScene<'_> {
    pub fn upload<'session>(
        session: &'session RenderSession,
        scene: &LoadedScene,
        progress: &mut dyn Progress,
    ) -> Result<GpuScene<'session>> {
        anyhow::ensure!(!progress.cancelled(), "render cancelled");
        let requirements = ShaderRequirements::new(
            &scene.file.materials,
            scene.file.aov_names.len(),
            scene.file.world_volume.as_ref(),
        )?;
        let scene_queries =
            requirements.scene_queries() || scene.file.materials.iter().any(|m| m.volume.is_some());
        let (shaders, ptx_time, pipeline_time) = session.shaders(requirements, progress)?;
        let raygen_pg = &shaders.raygen;
        let miss_rg_pg = &shaders.miss_radiance;
        let miss_shadow_pg = &shaders.miss_shadow;
        let hit_rg_pg = &shaders.hit_radiance;
        let hit_shadow_pg = &shaders.hit_shadow;
        let stream = session.stream()?;
        let uploads = UploadArena {
            stream: stream.clone(),
            buffers: Default::default(),
        };
        let render_settings = scene.file.render;
        let t_scene_upload = Instant::now();
        // --- Upload textures (shared across materials) ---
        let mut _tex_buffers: Vec<CudaSlice<f32>> = Vec::new();
        let tex_slots = principled::upload_textures(&scene.textures, &stream, &mut _tex_buffers)?;
        let texture_bytes: usize = _tex_buffers.iter().map(|b| b.len() * 4).sum();
        let source_bytes: usize = scene.textures.iter().map(|t| t.data().len() * 4).sum();
        progress.log(&format!(
            "GPU textures: {:.2} MiB (source {:.2} MiB)",
            texture_bytes as f64 / 1_048_576.0,
            source_bytes as f64 / 1_048_576.0,
        ));

        // --- Upload materials ---
        // Colour-graph buffers (node records + per-node LUT data) live for the
        // lifetime of the render; stash them so their CudaSlice drops are
        // deferred. Separate u32 / f32 vectors because RGBCurve LUTs are float.
        let mut graph_storage = principled::GraphStorage::default();
        // Volume parameter blocks live alongside materials. Each material that
        // has a Volume slot gets one allocation; the device-side `PrincipledGpu`
        // holds a pointer back to it (or null when absent).
        let mut mat_device_ptrs: Vec<optix_sys::CUdeviceptr> = Vec::new();
        // Recursively upload one material node (and its left/right binary-tree
        // children, when present) to the device, returning the device pointer
        // to its PrincipledGpu. Each call may allocate its own color graph
        // buffer + volume buffer + child sub-trees. The depth cap is a sanity
        // guard against producer bugs — even a fully balanced 4-leaf tree only
        // recurses 3 levels deep, so >8 is well past any artist intent.
        fn upload_one_mat(
            mat: &scene_format::PrincipledMaterial,
            tex_slots: &[TextureBinding],
            stream: &Arc<CudaStream>,
            graph_storage: &mut principled::GraphStorage,
            uploads: &UploadArena,
            depth: u32,
        ) -> Result<optix_sys::CUdeviceptr> {
            if depth > 8 {
                anyhow::bail!(
                    "PrincipledMaterial Mix tree deeper than 8 levels — likely a \
                 producer-side bug; refusing to recurse further"
                );
            }
            let graph = match &mat.color_graph {
                Some(g) => principled::upload_color_graph(
                    g,
                    tex_slots,
                    stream,
                    &mut graph_storage.nodes,
                    &mut graph_storage.tables,
                )?,
                None => principled::ColorGraphGpu::default(),
            };
            let volume_ptr = match &mat.volume {
                Some(v) => {
                    let mut vg = principled::make_volume_gpu(v);
                    if let Some(graph) = &v.emission_graph {
                        let binding = principled::upload_shader_graph(
                            graph,
                            tex_slots,
                            stream,
                            graph_storage,
                        )?;
                        vg.emission_nodes = binding.nodes;
                        vg.emission_len = binding.len;
                        vg.emission_contexts = binding.contexts;
                    }
                    uploads.copy(&vg)?
                }
                None => 0,
            };
            let secondary_ptr = match &mat.secondary {
                Some(sec) => {
                    upload_one_mat(sec, tex_slots, stream, graph_storage, uploads, depth + 1)?
                }
                None => 0,
            };
            let left_subtree_ptr = match &mat.left_subtree {
                Some(lhs) => {
                    upload_one_mat(lhs, tex_slots, stream, graph_storage, uploads, depth + 1)?
                }
                None => 0,
            };
            let mut gpu = principled::make_material_data(
                mat,
                tex_slots,
                graph,
                volume_ptr,
                secondary_ptr,
                left_subtree_ptr,
            );
            if let Some(graph) = &mat.shader_graph {
                let binding =
                    principled::upload_shader_graph(graph, tex_slots, stream, graph_storage)?;
                gpu.shader_nodes = binding.nodes;
                gpu.shader_len = binding.len;
                gpu.shader_outputs = binding.outputs;
                gpu.shader_contexts = binding.contexts;
            }
            let mut aov_shaders = Vec::new();
            for aov in &mat.aov_graphs {
                let binding =
                    principled::upload_shader_graph(&aov.graph, tex_slots, stream, graph_storage)?;
                let mut aov_gpu = gpu;
                aov_gpu.shader_nodes = binding.nodes;
                aov_gpu.shader_len = binding.len;
                aov_gpu.shader_outputs = binding.outputs;
                aov_gpu.shader_contexts = binding.contexts;
                aov_shaders.push(AovShader {
                    material: uploads.copy(&aov_gpu)?,
                    index: aov.index,
                    _pad: 0,
                });
            }
            if !aov_shaders.is_empty() {
                gpu.aov_shaders = uploads.copy_slice(&aov_shaders)?;
                gpu.num_aov_shaders = aov_shaders.len() as u32;
            }
            let (contexts, bump) = principled::surface_requirements(mat)?;
            gpu.surface_contexts = contexts;
            gpu.surface_bump = i32::from(bump);
            uploads.copy(&gpu)
        }
        for mat in &scene.file.materials {
            let p = upload_one_mat(mat, &tex_slots, &stream, &mut graph_storage, &uploads, 0)?;
            mat_device_ptrs.push(p);
        }
        // World volume sits in LaunchParams and is shared by every ray that
        // doesn't enter a tighter mesh-bounded volume. Allocated once.
        let world_volume_ptr: optix_sys::CUdeviceptr = match &scene.file.world_volume {
            Some(v) => {
                let mut vg = principled::make_volume_gpu(v);
                if let Some(graph) = &v.emission_graph {
                    let binding = principled::upload_shader_graph(
                        graph,
                        &tex_slots,
                        &stream,
                        &mut graph_storage,
                    )?;
                    vg.emission_nodes = binding.nodes;
                    vg.emission_len = binding.len;
                    vg.emission_contexts = binding.contexts;
                }
                uploads.copy(&vg)?
            }
            None => 0,
        };

        // --- Build per-mesh GAS ---
        let build_options = AccelBuildOptions {
            build_flags: BuildFlags::PREFER_FAST_TRACE,
            operation: BuildOperation::Build,
        };

        struct MeshGpu {
            query_nodes: u64,
            query_indices: u64,
            attributes: Vec<crate::gpu_types::ShaderAttribute>,
            handle: optix_sys::OptixTraversableHandle,
            d_verts: optix_sys::CUdeviceptr,
            d_normals: optix_sys::CUdeviceptr,
            d_indices: optix_sys::CUdeviceptr,
            d_uvs: optix_sys::CUdeviceptr,
            d_mat_indices: optix_sys::CUdeviceptr,
            d_vertex_colors: optix_sys::CUdeviceptr,
            d_vertex_alphas: optix_sys::CUdeviceptr,
            d_tangents: optix_sys::CUdeviceptr,
            num_verts: i32,
        }
        let mut _buffers: Vec<CudaSlice<u8>> = Vec::new();
        let mut scratch = Vec::new();
        let mut mesh_uploads = mesh_uploads::MeshUploads::new(stream.clone())?;
        let mut meshes_gpu: Vec<MeshGpu> = Vec::new();

        // Compare complete geometry against owned query data from the last render.
        // Cache misses still build in parallel batches of at most 32 meshes.
        session.begin_queries(if scene_queries { &scene.meshes } else { &[] });
        let meshes_with_queries =
            scene
                .meshes
                .chunks(32)
                .enumerate()
                .flat_map(|(batch, meshes)| {
                    let queries = if scene_queries {
                        session.query_batch(batch * 32, meshes)
                    } else {
                        let empty = Arc::new(query_cache::QueryTree::default());
                        vec![empty; meshes.len()]
                    };
                    meshes.iter().zip(queries)
                });
        for (m, query) in meshes_with_queries {
            let verts_ptr = mesh_uploads.copy_slice(&m.vertices[..])?;
            let idx_ptr = mesh_uploads.copy_slice(&m.indices[..])?;

            let vert_ptrs = [verts_ptr];
            let flags = [GeometryFlags::NONE];
            let num_verts = (m.vertices.len() / 3) as u32;
            let num_tris = (m.indices.len() / 3) as u32;

            let tri_input = TriangleArrayInput::new(
                &vert_ptrs,
                num_verts,
                VertexFormat::Float3,
                3 * mem::size_of::<f32>() as u32,
                &flags,
            )
            .with_indices(
                idx_ptr,
                num_tris,
                IndicesFormat::UnsignedInt3,
                3 * mem::size_of::<u32>() as u32,
            );

            let bi = [BuildInput::Triangles(tri_input)];
            // Wait only when GAS construction needs the position/index arrays.
            mesh_uploads.wait_on_consumer()?;
            let built = session
                .build_accel(&build_options, &bi)
                .context("mesh GAS")?;
            let handle = built.handle;
            let d_normals = mesh_uploads.copy_slice(&m.normals[..])?;
            let d_uvs = mesh_uploads.copy_slice(&m.uvs[..])?;
            let d_mat_indices = mesh_uploads.copy_slice(&m.material_indices[..])?;
            let d_vertex_colors = mesh_uploads.copy_slice(&m.vertex_colors[..])?;
            let d_vertex_alphas = mesh_uploads.copy_slice(&m.vertex_alphas[..])?;
            let d_tangents = mesh_uploads.copy_slice(&m.tangents[..])?;

            let mut attributes = Vec::new();
            for (key, values) in &m.attributes {
                let data = mesh_uploads.copy_slice(&values[..])?;
                attributes.push(crate::gpu_types::ShaderAttribute {
                    key: *key,
                    data,
                    count: (values.len() / 4) as u32,
                    _pad: 0,
                });
            }
            let query_nodes = mesh_uploads.copy_slice(&query.nodes)?;
            let query_indices = mesh_uploads.copy_slice(&query.indices)?;
            meshes_gpu.push(MeshGpu {
                query_nodes,
                query_indices,
                attributes,
                handle,
                d_verts: verts_ptr,
                d_normals,
                d_indices: idx_ptr,
                d_uvs,
                d_mat_indices,
                d_vertex_colors,
                d_vertex_alphas,
                d_tangents,
                num_verts: num_verts as i32,
            });

            scratch.push(built.scratch);
            _buffers.push(built.output);
        }
        mesh_uploads.finish()?;
        scratch.clear();

        // Reverse mapping object_idx → mesh_light_idx (or -1 when the object
        // isn't registered as a mesh-light). Built before the SBT loop so each
        // HitGroupData can carry the right `area_light_group`. The mesh-light
        // device upload itself happens later (after light gathering) — it
        // only consumes the table; doesn't write to it.
        let object_to_mesh_light: Vec<i32> = {
            let mut m = vec![-1i32; scene.objects.len()];
            for (i, ml) in scene.mesh_lights.iter().enumerate() {
                if let Some(slot) = m.get_mut(ml.object_idx as usize) {
                    *slot = i as i32;
                }
            }
            m
        };

        // --- Per-object SBT records: radiance and shadow ---
        let mut tri_hg_records: Vec<SbtRecord<HitGroupData>> = Vec::new();
        let mut query_objects = Vec::new();
        let mut query_bounds = Vec::new();
        let mut gas_entries: Vec<GasEntry> = Vec::new();
        let mut _mat_table_bufs: Vec<CudaSlice<u64>> = Vec::new();
        let mesh_bounds: Vec<_> = scene
            .meshes
            .iter()
            .map(|mesh| {
                let mut lo = [f32::MAX; 3];
                let mut hi = [f32::MIN; 3];
                for vertex in mesh.vertices.as_chunks::<3>().0 {
                    for axis in 0..3 {
                        lo[axis] = lo[axis].min(vertex[axis]);
                        hi[axis] = hi[axis].max(vertex[axis]);
                    }
                }
                (lo, hi)
            })
            .collect();
        for (obj_idx, obj) in scene.objects.iter().enumerate() {
            let m = &meshes_gpu[obj.mesh as usize];
            let mat_ptr = *mat_device_ptrs
                .get(obj.material as usize)
                .ok_or_else(|| anyhow::anyhow!("object {} material index out of range", obj_idx))?;

            // Per-object material table (used when the mesh has per-triangle IDs).
            let obj_materials = &scene.file.objects[obj_idx].materials;
            let (materials_ptr, num_materials) = if !obj_materials.is_empty() {
                let ptrs: Vec<u64> = obj_materials
                    .iter()
                    .map(|mi| match mat_device_ptrs.get(*mi as usize).copied() {
                        Some(p) => p as u64,
                        None => {
                            eprintln!(
                                "[vibrt] warn: object {} per-tri material index {} out of range \
                             (have {} materials) — falling back to object's primary material",
                                obj_idx,
                                mi,
                                mat_device_ptrs.len()
                            );
                            mat_ptr as u64
                        }
                    })
                    .collect();
                let slice = stream.clone_htod(&ptrs).cuda()?;
                let (p, _) = slice.device_ptr(&stream);
                _mat_table_bufs.push(slice);
                (p as optix_sys::CUdeviceptr, ptrs.len() as i32)
            } else {
                (0, 0)
            };

            let mut attributes = m.attributes.clone();
            for (name, value) in &scene.file.objects[obj_idx].attributes {
                let key = vibrt_nodes::shader::attribute_key(name);
                anyhow::ensure!(
                    !attributes.iter().any(|a| a.key == key),
                    "Duplicate object/mesh attribute {name:?}"
                );
                attributes.push(crate::gpu_types::ShaderAttribute {
                    key,
                    data: uploads.copy_slice(value)?,
                    count: 1,
                    _pad: 0,
                });
            }
            let hg = HitGroupData {
                attributes: if attributes.is_empty() {
                    0
                } else {
                    uploads.copy_slice(&attributes)?
                },
                num_attributes: attributes.len() as u32,
                mat: mat_ptr,
                vertices: m.d_verts,
                normals: m.d_normals,
                indices: m.d_indices,
                uvs: m.d_uvs,
                num_vertices: m.num_verts,
                // Mesh-light index for this instance, or -1 when the object
                // isn't registered for mesh-light NEE. The closest-hit shader
                // checks `>= 0` to enable MIS-weighting of BSDF-sampled
                // emission against the NEE sampler.
                area_light_group: object_to_mesh_light.get(obj_idx).copied().unwrap_or(-1),
                material_indices: m.d_mat_indices,
                materials: materials_ptr,
                num_materials,
                vertex_colors: m.d_vertex_colors,
                vertex_alphas: m.d_vertex_alphas,
                tangents: m.d_tangents,
                bounds_min: mesh_bounds[obj.mesh as usize].0,
                bounds_max: mesh_bounds[obj.mesh as usize].1,
                object_to_world: obj.transform,
                world_to_object: crate::transform::invert(&obj.transform),
                object_color: scene.file.objects[obj_idx].object_color,
                object_index: scene.file.objects[obj_idx].object_index,
                instance_id: obj_idx as u32,
                light_pdf: {
                    let total: f32 = scene.mesh_lights.iter().map(|l| l.power).sum();
                    if let Some(light) = scene
                        .mesh_lights
                        .iter()
                        .find(|l| l.object_idx as usize == obj_idx)
                    {
                        let mut densities =
                            vec![0.0f32; scene.meshes[obj.mesh as usize].indices.len() / 3];
                        for tri in &light.triangles {
                            densities[tri.primitive as usize] =
                                tri.sampling_luminance / total.max(1e-30);
                        }
                        uploads.copy_slice(&densities)?
                    } else {
                        0
                    }
                },
            };
            if scene_queries {
                // Local GPU queries index this table by HitGroupData::instance_id.
                // Only query_indices may be reordered by the top-level BVH build.
                debug_assert_eq!(query_objects.len(), obj_idx);
                let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
                for corner in 0..8 {
                    let p = std::array::from_fn(|a| {
                        if corner & (1 << a) == 0 {
                            hg.bounds_min[a]
                        } else {
                            hg.bounds_max[a]
                        }
                    });
                    let p = crate::transform::transform_point(&obj.transform, p);
                    for a in 0..3 {
                        lo[a] = lo[a].min(p[a]);
                        hi[a] = hi[a].max(p[a]);
                    }
                }
                query_bounds.push((lo, hi));
                query_objects.push(QueryObject {
                    geometry: 0,
                    nodes: m.query_nodes,
                    indices: m.query_indices,
                    mask: if scene.file.objects[obj_idx].cast_shadow {
                        7
                    } else {
                        5
                    },
                    _pad: 0,
                });
            }
            tri_hg_records.push(SbtRecord::new(hit_rg_pg, hg)?);
            tri_hg_records.push(SbtRecord::new(hit_shadow_pg, hg)?);
            gas_entries.push(GasEntry {
                handle: m.handle,
                sbt_offset: (obj_idx as u32) * 2,
                transform: obj.transform,
                cast_shadow: scene.file.objects[obj_idx].cast_shadow,
                camera_visible: scene.file.objects[obj_idx].camera_visible,
                // An instance can share a GAS with differently shaded instances.
                // Inspect every slot, including the fallback for invalid face IDs.
                opaque: std::iter::once(obj.material)
                    .chain(obj_materials.iter().copied())
                    .all(|id| {
                        let material = scene
                            .file
                            .materials
                            .get(id as usize)
                            .unwrap_or(&scene.file.materials[obj.material as usize]);
                        !principled::needs_any_hit(material)
                    }),
            });
        }

        // --- Build IAS ---
        if gas_entries.is_empty() {
            anyhow::bail!("no geometry in scene");
        }
        // Visibility mask bits (mirrored in `scene_loader::VIS_MASK_*`):
        //   0x01 = primary radiance (camera ray + specular bounce chain)
        //   0x02 = shadow ray (= "blocks NEE shadow rays from receivers"
        //          when AND-masked into the ray's mask)
        //   0x04 = secondary radiance (diffuse bounce continuation)
        // Camera-hidden emissive meshes drop the 0x01 bit so camera/specular
        // rays pass straight through, while still appearing on diffuse-bounce
        // rays + appearing as a NEE source via mesh-lights.
        let instances: Vec<optix_sys::OptixInstance> = gas_entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let mut mask: u32 = scene_loader::VIS_MASK_SECONDARY;
                if e.camera_visible {
                    mask |= scene_loader::VIS_MASK_PRIMARY;
                }
                if e.cast_shadow {
                    mask |= scene_loader::VIS_MASK_SHADOW;
                }
                optix_sys::OptixInstance {
                    transform: e.transform,
                    instanceId: i as u32,
                    sbtOffset: e.sbt_offset,
                    visibilityMask: mask,
                    flags: if e.opaque {
                        InstanceFlags::DISABLE_ANYHIT.bits()
                    } else {
                        InstanceFlags::NONE.bits()
                    },
                    traversableHandle: e.handle,
                    pad: [0; 2],
                }
            })
            .collect();
        let d_instances = uploads.copy_slice(&instances)?;
        let ias_input = [BuildInput::Instances(accel::InstanceArrayInput {
            instances: d_instances,
            num_instances: instances.len() as u32,
        })];
        let built = session
            .build_accel(&build_options, &ias_input)
            .context("scene IAS")?;
        let traversable = built.handle;
        stream.synchronize().cuda()?;
        drop(built.scratch);
        _buffers.push(built.output);

        // --- SBT ---
        let raygen_record = SbtRecord::new(raygen_pg, RayGenData {})?;
        let miss_rg_record = SbtRecord::new(miss_rg_pg, MissData { _unused: 0 })?;
        let miss_shadow_record = SbtRecord::new(miss_shadow_pg, MissData { _unused: 0 })?;

        let d_rg = uploads.copy(&raygen_record)?;
        let mut miss_bytes: Vec<u8> = Vec::new();
        let miss_stride = mem::size_of_val(&miss_rg_record);
        miss_bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&miss_rg_record as *const _ as *const u8, miss_stride)
        });
        miss_bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&miss_shadow_record as *const _ as *const u8, miss_stride)
        });
        let d_miss = uploads.copy_slice(&miss_bytes)?;

        let hg_stride = mem::size_of::<SbtRecord<HitGroupData>>();
        let mut hg_bytes: Vec<u8> = Vec::new();
        for rec in &tri_hg_records {
            hg_bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(rec as *const _ as *const u8, hg_stride)
            });
        }
        let d_hg = uploads.copy_slice(&hg_bytes)?;
        let (query_nodes, query_indices, query_objects_ptr) =
            if scene_queries && !query_objects.is_empty() {
                for (i, obj) in query_objects.iter_mut().enumerate() {
                    obj.geometry = d_hg
                        + (i * 2 * hg_stride + std::mem::offset_of!(SbtRecord<HitGroupData>, data))
                            as u64;
                }
                let (nodes, indices) = crate::query_bvh::build(&query_bounds);
                (
                    uploads.copy_slice(&nodes)?,
                    uploads.copy_slice(&indices)?,
                    uploads.copy_slice(&query_objects)?,
                )
            } else {
                (0, 0, 0)
            };

        // Only bounded media need per-camera containment queries. Reuse their
        // geometry BVHs; do not trace every object for every camera sample.
        let initial_volumes: Vec<_> = query_objects
            .iter()
            .enumerate()
            .filter_map(|(i, q)| {
                let object = &scene.file.objects[i];
                std::iter::once(&object.material)
                    .chain(object.materials.iter())
                    .any(|m| {
                        scene
                            .file
                            .materials
                            .get(*m as usize)
                            .is_some_and(|m| m.volume.is_some())
                    })
                    .then_some(*q)
            })
            .collect();
        let initial_volumes_ptr = uploads.copy_slice(&initial_volumes)?;

        let sbt_builder = ShaderBindingTableBuilder::new(d_rg)
            .miss_records(d_miss, miss_stride as u32, 2)
            .hitgroup_records(d_hg, hg_stride as u32, tri_hg_records.len() as u32);
        let sbt = sbt_builder.build()?;

        // --- Lights ---
        // IES profiles must be uploaded BEFORE the light vectors are
        // copied to the device, so each light struct carries its IES
        // device pointer when it lands on the GPU. Patch the host-side
        // copies in place; the upstream (LoadedScene) is consumed below
        // anyway. `_d_ies_*` keep the device buffers alive for the
        // lifetime of the render — drop on render exit.
        let mut point_lights = scene.point_lights.clone();
        let mut spot_lights = scene.spot_lights.clone();
        for shader in point_lights
            .iter_mut()
            .map(|l| &mut l.shader)
            .chain(spot_lights.iter_mut().map(|l| &mut l.shader))
        {
            if *shader != 0 {
                *shader = *mat_device_ptrs
                    .get((*shader - 1) as usize)
                    .context("light shader material index out of range")?;
            }
        }
        let mut rect_lights = scene.rect_lights.clone();
        for light in &mut rect_lights {
            if light.shader != 0 {
                light.shader = *mat_device_ptrs
                    .get((light.shader - 1) as usize)
                    .context("area-light shader material index out of range")?;
            }
        }
        let _d_ies_point = upload_ies_buffers(&uploads, &mut point_lights, &scene.point_ies)?;
        let _d_ies_spot = upload_ies_buffers(&uploads, &mut spot_lights, &scene.spot_ies)?;
        let _d_ies_rect = upload_ies_buffers(&uploads, &mut rect_lights, &scene.rect_ies)?;
        let d_points = uploads.copy_slice(&point_lights)?;
        let d_suns = uploads.copy_slice(&scene.sun_lights)?;
        let d_spots = uploads.copy_slice(&spot_lights)?;
        let d_rects = uploads.copy_slice(&rect_lights)?;

        // Mesh lights: convert each host triangle into the GPU `MeshLightTri`
        // POD, resolving `emission_tex_idx` to the device-side texture
        // pointer/dimensions from `tex_slots`. Then upload per-triangle
        // table + inner CDF, copy the outer `MeshLight` array, and build
        // the outer CDF over per-light total power.
        let mut mesh_lights_gpu: Vec<MeshLight> = Vec::with_capacity(scene.mesh_lights.len());
        let mut mesh_light_outer_cdf = vec![0.0f32; scene.mesh_lights.len() + 1];
        for (i, ml) in scene.mesh_lights.iter().enumerate() {
            let gpu_tris: Vec<MeshLightTri> = ml
                .triangles
                .iter()
                .map(|t| {
                    let (tex_ptr, tex_w, tex_h, tex_c) = match t.emission_tex_idx {
                        Some(idx) => tex_slots.get(idx as usize).copied().unwrap_or_default(),
                        None => (0, 0, 0, 0),
                    };
                    MeshLightTri {
                        geometry: d_hg
                            + (ml.object_idx as usize * 2 * hg_stride
                                + mem::offset_of!(SbtRecord<HitGroupData>, data))
                                as u64,
                        primitive: t.primitive,
                        _pad_geometry: 0,
                        v0: t.v0,
                        _pad0: 0.0,
                        v1: t.v1,
                        _pad1: 0.0,
                        v2: t.v2,
                        area: t.area,
                        emission: t.emission,
                        _pad2: 0.0,
                        uv0: t.uv0,
                        uv1: t.uv1,
                        uv2: t.uv2,
                        _pad3: [0.0, 0.0],
                        emission_tex: tex_ptr,
                        emission_tex_w: tex_w,
                        emission_tex_h: tex_h,
                        emission_tex_channels: tex_c,
                        _pad4: [0, 0, 0],
                    }
                })
                .collect();
            let d_tris = uploads.copy_slice(&gpu_tris)?;
            let d_cdf = uploads.copy_slice(&ml.cdf)?;
            mesh_lights_gpu.push(MeshLight {
                triangles: d_tris,
                cdf: d_cdf,
                num_triangles: ml.triangles.len() as u32,
                two_sided: if ml.two_sided { 1 } else { 0 },
                power: ml.power,
                _pad: 0,
            });
            mesh_light_outer_cdf[i + 1] = mesh_light_outer_cdf[i] + ml.power.max(0.0);
        }
        let mesh_light_total_power = *mesh_light_outer_cdf.last().unwrap_or(&0.0);
        if mesh_light_total_power > 0.0 {
            let inv = 1.0 / mesh_light_total_power;
            for v in mesh_light_outer_cdf.iter_mut().skip(1) {
                *v *= inv;
            }
        }
        let d_mesh_lights = if mesh_lights_gpu.is_empty() {
            0
        } else {
            uploads.copy_slice(&mesh_lights_gpu)?
        };
        let d_mesh_light_cdf = if mesh_lights_gpu.is_empty() {
            0
        } else {
            uploads.copy_slice(&mesh_light_outer_cdf)?
        };

        // --- World / envmap ---
        let (world_type, world_color, world_strength) = match &scene.file.world {
            Some(scene_format::WorldDesc::Constant { color, strength }) => (0, *color, *strength),
            Some(scene_format::WorldDesc::Envmap { strength, .. }) => (1, [0.0; 3], *strength),
            // For mixed worlds the per-layer strengths live on each
            // EnvmapLayer; world_strength is unused on the kernel side
            // (world_background takes the layer mix).
            Some(scene_format::WorldDesc::Mixed { .. }) => (2, [0.0; 3], 1.0),
            None => (0, [0.0; 3], 0.0),
        };
        let envmap_rot = match &scene.file.world {
            Some(scene_format::WorldDesc::Envmap { rotation_z_rad, .. }) => *rotation_z_rad,
            _ => 0.0,
        };
        let (d_envmap, env_w, env_h, d_mcdf, d_ccdf, env_integral) =
            if let Some((rgb, w, h)) = &scene.envmap_rgb {
                let d_env = uploads.copy_slice(rgb)?;
                let (m, c, total) = super::lighting::build_envmap_cdf(rgb, *w, *h);
                let d_m = uploads.copy_slice(&m)?;
                let d_c = uploads.copy_slice(&c)?;
                (d_env, *w as i32, *h as i32, d_m, d_c, total)
            } else {
                (0, 0, 0, 0, 0, 0.0)
            };
        // Mixed-envmap layers (uploaded only when WorldDesc::Mixed).
        let (d_envmap_a, env_w_a, env_h_a) = if let Some((rgb, w, h)) = &scene.envmap_layer_a_rgb {
            let d = uploads.copy_slice(rgb)?;
            (d, *w as i32, *h as i32)
        } else {
            (0, 0, 0)
        };
        let (d_envmap_b, env_w_b, env_h_b) = if let Some((rgb, w, h)) = &scene.envmap_layer_b_rgb {
            let d = uploads.copy_slice(rgb)?;
            (d, *w as i32, *h as i32)
        } else {
            (0, 0, 0)
        };
        let (env_strength_a, env_strength_b, env_rot_a, env_rot_b, env_mix_fac, env_split_camera) =
            match &scene.file.world {
                Some(scene_format::WorldDesc::Mixed {
                    a,
                    b,
                    fac,
                    split_by_camera_ray,
                }) => (
                    a.strength,
                    b.strength,
                    a.rotation,
                    b.rotation,
                    *fac,
                    if *split_by_camera_ray { 1 } else { 0 },
                ),
                _ => (
                    0.0,
                    0.0,
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    0.0,
                    0,
                ),
            };

        let params = LaunchParams {
            query_nodes,
            query_indices,
            query_objects: query_objects_ptr,
            initial_volumes: initial_volumes_ptr,
            num_initial_volumes: initial_volumes.len() as u32,
            aov_data: 0,
            num_aovs: scene.file.aov_names.len() as u32,
            image: 0,
            accumulation: 0,
            width: render_settings.width,
            height: render_settings.height,
            samples_per_pixel: render_settings.spp,
            sample_start: 0,
            sample_count: 1,
            film_transparent: render_settings.film_transparent as u32,
            max_depth: render_settings.max_depth,
            max_diffuse_bounces: render_settings.max_diffuse_bounces,
            max_glossy_bounces: render_settings.max_glossy_bounces,
            max_transmission_bounces: render_settings.max_transmission_bounces,
            max_transparent_bounces: render_settings.max_transparent_bounces,
            cam_eye: [0.0; 3],
            cam_u: [0.0; 3],
            cam_v: [0.0; 3],
            cam_w: [0.0; 3],
            cam_lens_radius: scene.file.camera.lens_radius,
            cam_focal_distance: scene.file.camera.focal_distance,
            cam_clip_start: scene.file.camera.clip_start,
            cam_clip_end: scene.file.camera.clip_end,
            cam_shift_x: scene.file.camera.shift_x,
            cam_shift_y: scene.file.camera.shift_y,
            cam_orthographic: u32::from(scene.file.camera.ortho_height > 0.0),
            traversable,
            num_point_lights: scene.point_lights.len() as i32,
            point_lights: d_points,
            num_sun_lights: scene.sun_lights.len() as i32,
            sun_lights: d_suns,
            num_spot_lights: scene.spot_lights.len() as i32,
            spot_lights: d_spots,
            num_rect_lights: scene.rect_lights.len() as i32,
            rect_lights: d_rects,
            num_mesh_lights: scene.mesh_lights.len() as i32,
            mesh_lights: d_mesh_lights,
            mesh_light_cdf: d_mesh_light_cdf,
            mesh_light_total_power,
            world_type,
            world_color,
            world_strength,
            envmap_data: d_envmap,
            envmap_width: env_w,
            envmap_height: env_h,
            envmap_marginal_cdf: d_mcdf,
            envmap_conditional_cdf: d_ccdf,
            envmap_integral: env_integral,
            envmap_rotation_z_rad: envmap_rot,
            clamp_indirect: render_settings.clamp_indirect,
            clamp_direct: render_settings.clamp_direct,
            filter_glossy: render_settings.filter_glossy,
            envmap_data_a: d_envmap_a,
            envmap_width_a: env_w_a,
            envmap_height_a: env_h_a,
            envmap_data_b: d_envmap_b,
            envmap_width_b: env_w_b,
            envmap_height_b: env_h_b,
            envmap_strength_a: env_strength_a,
            envmap_strength_b: env_strength_b,
            envmap_rotation_a: env_rot_a,
            envmap_rotation_b: env_rot_b,
            envmap_mix_fac: env_mix_fac,
            envmap_split_by_camera_ray: env_split_camera,
            albedo_aov: 0,
            normal_aov: 0,
            depth_aov: 0,
            world_volume: world_volume_ptr,
            primary_back_face_skip_exhausted: 0,
        };

        stream.synchronize().cuda()?;
        Ok(GpuScene {
            session,
            shaders,
            params,
            sbt,
            camera: scene.file.camera.clone(),
            settings: scene.file.render,
            aov_names: scene.file.aov_names.clone(),
            ptx_time,
            pipeline_time,
            upload_time: t_scene_upload.elapsed(),
            _uploads: uploads,
            _textures: _tex_buffers,
            _graphs: graph_storage,
            _accels: _buffers,
            _meshes: mesh_uploads,
            _material_tables: _mat_table_bufs,
        })
    }
}

impl Drop for GpuScene<'_> {
    fn drop(&mut self) {
        self.session.cleanup();
    }
}

/// Light types that carry an optional IES profile. The trait gives
/// `upload_ies_buffers` a uniform way to patch each light's
/// `ies_data` / `ies_n_v` / `ies_n_h` fields without three near-
/// identical helpers.
trait HasIes {
    fn set_ies(&mut self, data: optix_sys::CUdeviceptr, n_v: u32, n_h: u32);
}
impl HasIes for crate::gpu_types::PointLight {
    fn set_ies(&mut self, data: optix_sys::CUdeviceptr, n_v: u32, n_h: u32) {
        self.ies_data = data;
        self.ies_n_v = n_v;
        self.ies_n_h = n_h;
    }
}
impl HasIes for crate::gpu_types::SpotLight {
    fn set_ies(&mut self, data: optix_sys::CUdeviceptr, n_v: u32, n_h: u32) {
        self.ies_data = data;
        self.ies_n_v = n_v;
        self.ies_n_h = n_h;
    }
}
impl HasIes for crate::gpu_types::AreaRectLight {
    fn set_ies(&mut self, data: optix_sys::CUdeviceptr, n_v: u32, n_h: u32) {
        self.ies_data = data;
        self.ies_n_v = n_v;
        self.ies_n_h = n_h;
    }
}

/// Pack each `Some(IesProfile)` into a flat float buffer
/// `[thetas (n_v) | phis (n_h) | normalised candelas (n_v * n_h)]`,
/// upload to the device, and patch the matching light's IES fields
/// to point at it. Returns the device pointers so the caller can keep
/// the allocations alive for the lifetime of the render.
///
/// The packing is what `ies_lookup` in devicecode.cu reads; keep the
/// two in sync.
fn upload_ies_buffers<L: HasIes>(
    uploads: &UploadArena,
    lights: &mut [L],
    profiles: &[Option<scene_format::IesProfile>],
) -> Result<Vec<optix_sys::CUdeviceptr>> {
    let mut out = Vec::new();
    for (i, profile) in profiles.iter().enumerate() {
        let Some(p) = profile else {
            continue;
        };
        let n_v = p.thetas_deg.len();
        let n_h = p.phis_deg.len().max(1);
        if n_v == 0 || p.candelas.len() != n_v * p.phis_deg.len() {
            // Malformed profile — silently skip rather than crashing
            // mid-render. The exporter is supposed to validate; this
            // is just defence in depth.
            continue;
        }
        let peak = p.peak_candela.max(1e-6);
        let mut buf: Vec<f32> = Vec::with_capacity(n_v + n_h + n_v * n_h);
        buf.extend_from_slice(&p.thetas_deg);
        // Synthesise a single phi=0 entry for radially-symmetric
        // profiles so the GPU lookup can read the phi axis without
        // a special case for n_h==1.
        if p.phis_deg.is_empty() {
            buf.push(0.0);
        } else {
            buf.extend_from_slice(&p.phis_deg);
        }
        for c in &p.candelas {
            buf.push(c / peak);
        }
        let d = uploads.copy_slice(&buf)?;
        lights[i].set_ies(d, n_v as u32, n_h as u32);
        out.push(d);
    }
    Ok(out)
}

struct GasEntry {
    handle: optix_sys::OptixTraversableHandle,
    sbt_offset: u32,
    transform: [f32; 12],
    cast_shadow: bool,
    camera_visible: bool,
    opaque: bool,
}

impl GpuScene<'_> {
    pub(super) fn stream(&self) -> Result<Arc<CudaStream>> {
        self.session.stream()
    }

    pub(super) fn denoiser(&self) -> Result<optix::denoiser::Denoiser> {
        self.session.denoiser()
    }

    pub(super) fn launch_params(
        &self,
        render_settings: scene_format::RenderSettings,
    ) -> LaunchParams {
        // --- Camera ---
        let aspect = if self.camera.frame_aspect > 0.0 {
            self.camera.frame_aspect
        } else {
            render_settings.width as f32 / render_settings.height as f32
        };
        let (eye, cam_u, cam_v, cam_w) = camera::compute_camera(
            &self.camera.transform,
            self.camera.fov_y_rad,
            aspect,
            self.camera.ortho_height,
        );
        let mut lp = self.params;
        lp.cam_eye = eye;
        lp.cam_u = cam_u;
        lp.cam_v = cam_v;
        lp.cam_w = cam_w;
        lp.width = render_settings.width;
        lp.height = render_settings.height;
        lp.samples_per_pixel = render_settings.spp;
        lp.film_transparent = render_settings.film_transparent as u32;
        lp.max_depth = render_settings.max_depth;
        lp.max_diffuse_bounces = render_settings.max_diffuse_bounces;
        lp.max_glossy_bounces = render_settings.max_glossy_bounces;
        lp.max_transmission_bounces = render_settings.max_transmission_bounces;
        lp.max_transparent_bounces = render_settings.max_transparent_bounces;
        lp.clamp_indirect = render_settings.clamp_indirect;
        lp.clamp_direct = render_settings.clamp_direct;
        lp.filter_glossy = render_settings.filter_glossy;
        lp
    }

    /// The scene retains the pipeline, SBT and all geometry until queued launches finish.
    /// The caller must keep the launch parameters and output buffers alive too.
    pub(super) unsafe fn launch(
        &self,
        stream: &CudaStream,
        params: optix_sys::CUdeviceptr,
        width: u32,
        height: u32,
    ) -> Result<()> {
        self.shaders
            .pipeline
            .launch(
                stream.cu_stream() as optix_sys::CUstream,
                params,
                mem::size_of::<LaunchParams>(),
                &self.sbt,
                width,
                height,
                1,
            )
            .context("launch")
    }

    pub(super) fn aov_count(&self) -> usize {
        self.aov_names.len()
    }

    pub(super) fn collect_aovs(
        &self,
        values: &[f32],
        pixel_count: usize,
        spp: u32,
    ) -> std::collections::BTreeMap<String, Vec<f32>> {
        self.aov_names
            .iter()
            .zip(values.chunks_exact(pixel_count * 4))
            .map(|(name, pixels)| {
                (
                    name.clone(),
                    pixels.iter().map(|v| v / spp as f32).collect(),
                )
            })
            .collect()
    }
}
