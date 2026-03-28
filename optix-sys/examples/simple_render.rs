use optix_sys::cuda::{CudaApi, CUDA_SUCCESS};
use optix_sys::*;
use std::ffi::c_void;
use std::mem;
use std::ptr;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;

// Must match devicecode.h layout exactly
#[repr(C)]
struct Params {
    image: CUdeviceptr,
    image_width: u32,
    image_height: u32,
    cam_eye: [f32; 3],
    cam_u: [f32; 3],
    cam_v: [f32; 3],
    cam_w: [f32; 3],
    handle: OptixTraversableHandle,
}

// SBT record: header (32 bytes) + data
#[repr(C, align(16))]
struct SbtRecord<T: Copy> {
    header: [u8; OPTIX_SBT_RECORD_HEADER_SIZE],
    data: T,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct RayGenData {}

#[repr(C)]
#[derive(Copy, Clone)]
struct MissData {
    bg_color: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct HitGroupData {}

macro_rules! cuda_check {
    ($cuda:expr, $call:expr) => {{
        let result = $call;
        if result != CUDA_SUCCESS {
            panic!("CUDA error: {} returned {}", stringify!($call), result);
        }
    }};
}

macro_rules! optix_check {
    ($call:expr) => {{
        let result = $call;
        if result != OptixResult::OPTIX_SUCCESS {
            panic!("OptiX error: {} returned {:?}", stringify!($call), result.0);
        }
    }};
}

fn main() {
    // --- Load CUDA and OptiX ---
    let cu = CudaApi::load().expect("Failed to load CUDA");
    let optix = optix_init().expect("Failed to initialize OptiX");

    unsafe {
        // --- Initialize CUDA ---
        cuda_check!(cu, (cu.cuInit)(0));
        let mut cu_device: i32 = 0;
        cuda_check!(cu, (cu.cuDeviceGet)(&mut cu_device, 0));
        let mut cu_ctx: CUcontext = ptr::null_mut();
        cuda_check!(cu, (cu.cuCtxCreate_v2)(&mut cu_ctx, 0, cu_device));

        let mut stream: CUstream = ptr::null_mut();
        cuda_check!(cu, (cu.cuStreamCreate)(&mut stream, 0));

        // --- Create OptiX device context ---
        let mut ctx: OptixDeviceContext = ptr::null_mut();
        let ctx_options = OptixDeviceContextOptions::default();
        optix_check!((optix.optixDeviceContextCreate.unwrap())(
            cu_ctx,
            &ctx_options,
            &mut ctx,
        ));

        // --- Load PTX module ---
        let ptx_path = std::path::Path::new(file!()).parent().unwrap().join("devicecode.ptx");
        let ptx = std::fs::read_to_string(&ptx_path)
            .unwrap_or_else(|_| panic!(
                "Failed to load {}. Compile it first:\n  \
                 nvcc -ptx examples/devicecode.cu -o examples/devicecode.ptx \
                 -I\"<OptiX SDK>/include\" -Iexamples --use_fast_math",
                ptx_path.display()
            ));
        let ptx_cstr = std::ffi::CString::new(ptx.as_str()).unwrap();

        let module_options = OptixModuleCompileOptions {
            maxRegisterCount: OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT as i32,
            optLevel: OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel: OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            ..Default::default()
        };

        let pipeline_options = OptixPipelineCompileOptions {
            usesMotionBlur: 0,
            traversableGraphFlags:
                OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS.0
                    as u32,
            numPayloadValues: 3,
            numAttributeValues: 3,
            exceptionFlags: OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE.0 as u32,
            pipelineLaunchParamsVariableName: b"params\0".as_ptr() as *const i8,
            ..Default::default()
        };

        let mut module: OptixModule = ptr::null_mut();
        let mut log = [0u8; 2048];
        let mut log_size = log.len();
        optix_check!((optix.optixModuleCreate.unwrap())(
            ctx,
            &module_options,
            &pipeline_options,
            ptx_cstr.as_ptr(),
            ptx.len(),
            log.as_mut_ptr() as *mut i8,
            &mut log_size,
            &mut module,
        ));

        // --- Create program groups ---
        let mut raygen_pg: OptixProgramGroup = ptr::null_mut();
        let mut miss_pg: OptixProgramGroup = ptr::null_mut();
        let mut hitgroup_pg: OptixProgramGroup = ptr::null_mut();
        let pg_options = OptixProgramGroupOptions::default();

        // Raygen
        {
            let desc = OptixProgramGroupDesc {
                kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                __bindgen_anon_1: OptixProgramGroupDesc__bindgen_ty_1 {
                    raygen: OptixProgramGroupSingleModule {
                        module,
                        entryFunctionName: b"__raygen__rg\0".as_ptr() as *const i8,
                    },
                },
                flags: 0,
            };
            log_size = log.len();
            optix_check!((optix.optixProgramGroupCreate.unwrap())(
                ctx,
                &desc,
                1,
                &pg_options,
                log.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut raygen_pg,
            ));
        }

        // Miss
        {
            let desc = OptixProgramGroupDesc {
                kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS,
                __bindgen_anon_1: OptixProgramGroupDesc__bindgen_ty_1 {
                    miss: OptixProgramGroupSingleModule {
                        module,
                        entryFunctionName: b"__miss__ms\0".as_ptr() as *const i8,
                    },
                },
                flags: 0,
            };
            log_size = log.len();
            optix_check!((optix.optixProgramGroupCreate.unwrap())(
                ctx,
                &desc,
                1,
                &pg_options,
                log.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut miss_pg,
            ));
        }

        // Hit group (closest hit)
        {
            let mut hitgroup = OptixProgramGroupHitgroup::default();
            hitgroup.moduleCH = module;
            hitgroup.entryFunctionNameCH = b"__closesthit__ch\0".as_ptr() as *const i8;

            let desc = OptixProgramGroupDesc {
                kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                __bindgen_anon_1: OptixProgramGroupDesc__bindgen_ty_1 { hitgroup },
                flags: 0,
            };
            log_size = log.len();
            optix_check!((optix.optixProgramGroupCreate.unwrap())(
                ctx,
                &desc,
                1,
                &pg_options,
                log.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut hitgroup_pg,
            ));
        }

        // --- Create pipeline ---
        let link_options = OptixPipelineLinkOptions {
            maxTraceDepth: 1,
        };
        let program_groups = [raygen_pg, miss_pg, hitgroup_pg];
        let mut pipeline: OptixPipeline = ptr::null_mut();
        log_size = log.len();
        optix_check!((optix.optixPipelineCreate.unwrap())(
            ctx,
            &pipeline_options,
            &link_options,
            program_groups.as_ptr(),
            program_groups.len() as u32,
            log.as_mut_ptr() as *mut i8,
            &mut log_size,
            &mut pipeline,
        ));

        // Set stack sizes
        optix_check!((optix.optixPipelineSetStackSize.unwrap())(
            pipeline,
            2 * 1024, // direct callable from traversal
            2 * 1024, // direct callable from state
            2 * 1024, // continuation
            1,        // max traversable graph depth
        ));

        // --- Build acceleration structure (single triangle) ---
        let vertices: [[f32; 3]; 3] = [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
        ];

        let mut d_vertices: CUdeviceptr = 0;
        let vertices_size = mem::size_of_val(&vertices);
        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_vertices, vertices_size));
        // Upload vertices — we need cuMemcpyHtoD, let's use a simple approach
        // Actually we need cuMemcpyHtoD which isn't in our minimal API. Let's add a workaround.
        upload_to_device(&cu, d_vertices, vertices.as_ptr() as *const c_void, vertices_size);

        let triangle_input = OptixBuildInputTriangleArray {
            vertexBuffers: &d_vertices as *const CUdeviceptr,
            numVertices: 3,
            vertexFormat: OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3,
            vertexStrideInBytes: 3 * mem::size_of::<f32>() as u32,
            ..Default::default()
        };

        let flags = OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_NONE.0 as u32;
        // Need to set the flags pointer
        let mut triangle_input = triangle_input;
        triangle_input.flags = &flags;
        triangle_input.numSbtRecords = 1;

        let build_input = OptixBuildInput {
            type_: OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
            __bindgen_anon_1: OptixBuildInput__bindgen_ty_1 {
                triangleArray: triangle_input,
            },
        };

        let accel_options = OptixAccelBuildOptions {
            buildFlags: OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION.0 as u32,
            operation: OptixBuildOperation::OPTIX_BUILD_OPERATION_BUILD,
            ..Default::default()
        };

        let mut buffer_sizes = OptixAccelBufferSizes::default();
        optix_check!((optix.optixAccelComputeMemoryUsage.unwrap())(
            ctx,
            &accel_options,
            &build_input,
            1,
            &mut buffer_sizes,
        ));

        let mut d_temp: CUdeviceptr = 0;
        let mut d_output: CUdeviceptr = 0;
        cuda_check!(
            cu,
            (cu.cuMemAlloc_v2)(&mut d_temp, buffer_sizes.tempSizeInBytes)
        );
        cuda_check!(
            cu,
            (cu.cuMemAlloc_v2)(&mut d_output, buffer_sizes.outputSizeInBytes)
        );

        let mut gas_handle: OptixTraversableHandle = 0;
        optix_check!((optix.optixAccelBuild.unwrap())(
            ctx,
            stream,
            &accel_options,
            &build_input,
            1,
            d_temp,
            buffer_sizes.tempSizeInBytes,
            d_output,
            buffer_sizes.outputSizeInBytes,
            &mut gas_handle,
            ptr::null(),
            0,
        ));
        cuda_check!(cu, (cu.cuStreamSynchronize)(stream));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_temp));

        // --- Build SBT ---
        let mut raygen_record = SbtRecord::<RayGenData> {
            header: [0u8; OPTIX_SBT_RECORD_HEADER_SIZE],
            data: RayGenData {},
        };
        optix_check!((optix.optixSbtRecordPackHeader.unwrap())(
            raygen_pg,
            raygen_record.header.as_mut_ptr() as *mut c_void,
        ));

        let mut miss_record = SbtRecord::<MissData> {
            header: [0u8; OPTIX_SBT_RECORD_HEADER_SIZE],
            data: MissData {
                bg_color: [0.1, 0.1, 0.3], // dark blue background
            },
        };
        optix_check!((optix.optixSbtRecordPackHeader.unwrap())(
            miss_pg,
            miss_record.header.as_mut_ptr() as *mut c_void,
        ));

        let mut hitgroup_record = SbtRecord::<HitGroupData> {
            header: [0u8; OPTIX_SBT_RECORD_HEADER_SIZE],
            data: HitGroupData {},
        };
        optix_check!((optix.optixSbtRecordPackHeader.unwrap())(
            hitgroup_pg,
            hitgroup_record.header.as_mut_ptr() as *mut c_void,
        ));

        // Upload SBT records to device
        let mut d_raygen_record: CUdeviceptr = 0;
        let mut d_miss_record: CUdeviceptr = 0;
        let mut d_hitgroup_record: CUdeviceptr = 0;

        let rg_size = mem::size_of_val(&raygen_record);
        let ms_size = mem::size_of_val(&miss_record);
        let hg_size = mem::size_of_val(&hitgroup_record);

        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_raygen_record, rg_size));
        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_miss_record, ms_size));
        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_hitgroup_record, hg_size));

        upload_to_device(
            &cu,
            d_raygen_record,
            &raygen_record as *const _ as *const c_void,
            rg_size,
        );
        upload_to_device(
            &cu,
            d_miss_record,
            &miss_record as *const _ as *const c_void,
            ms_size,
        );
        upload_to_device(
            &cu,
            d_hitgroup_record,
            &hitgroup_record as *const _ as *const c_void,
            hg_size,
        );

        let sbt = OptixShaderBindingTable {
            raygenRecord: d_raygen_record,
            exceptionRecord: 0,
            missRecordBase: d_miss_record,
            missRecordStrideInBytes: ms_size as u32,
            missRecordCount: 1,
            hitgroupRecordBase: d_hitgroup_record,
            hitgroupRecordStrideInBytes: hg_size as u32,
            hitgroupRecordCount: 1,
            callablesRecordBase: 0,
            callablesRecordStrideInBytes: 0,
            callablesRecordCount: 0,
        };

        // --- Allocate output image ---
        let image_size = (WIDTH * HEIGHT) as usize * mem::size_of::<u32>();
        let mut d_image: CUdeviceptr = 0;
        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_image, image_size));

        // --- Set up camera ---
        let cam_eye = [0.0f32, 0.0, 2.0];
        let cam_u = [1.2f32, 0.0, 0.0]; // right
        let cam_v = [0.0f32, 0.9, 0.0]; // up
        let cam_w = [0.0f32, 0.0, -1.0]; // forward (into screen)

        let params = Params {
            image: d_image,
            image_width: WIDTH,
            image_height: HEIGHT,
            cam_eye,
            cam_u,
            cam_v,
            cam_w,
            handle: gas_handle,
        };

        // Upload params to device
        let mut d_params: CUdeviceptr = 0;
        let params_size = mem::size_of::<Params>();
        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_params, params_size));
        upload_to_device(
            &cu,
            d_params,
            &params as *const Params as *const c_void,
            params_size,
        );

        // --- Launch ---
        println!("Launching OptiX render ({} x {})...", WIDTH, HEIGHT);
        optix_check!((optix.optixLaunch.unwrap())(
            pipeline,
            stream,
            d_params,
            params_size,
            &sbt,
            WIDTH,
            HEIGHT,
            1,
        ));
        cuda_check!(cu, (cu.cuStreamSynchronize)(stream));

        // --- Download image ---
        let mut pixels = vec![0u32; (WIDTH * HEIGHT) as usize];
        cuda_check!(
            cu,
            (cu.cuMemcpyDtoH_v2)(
                pixels.as_mut_ptr() as *mut c_void,
                d_image,
                image_size,
            )
        );

        // --- Save as PPM (no external dependency needed) ---
        save_ppm("output.ppm", WIDTH, HEIGHT, &pixels);
        println!("Saved output.ppm ({} x {})", WIDTH, HEIGHT);

        // --- Cleanup ---
        cuda_check!(cu, (cu.cuMemFree_v2)(d_image));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_params));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_raygen_record));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_miss_record));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_hitgroup_record));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_output));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_vertices));

        (optix.optixPipelineDestroy.unwrap())(pipeline);
        (optix.optixProgramGroupDestroy.unwrap())(raygen_pg);
        (optix.optixProgramGroupDestroy.unwrap())(miss_pg);
        (optix.optixProgramGroupDestroy.unwrap())(hitgroup_pg);
        (optix.optixModuleDestroy.unwrap())(module);
        (optix.optixDeviceContextDestroy.unwrap())(ctx);

        cuda_check!(cu, (cu.cuStreamDestroy_v2)(stream));
        cuda_check!(cu, (cu.cuCtxDestroy_v2)(cu_ctx));
    }
}

unsafe fn upload_to_device(cu: &CudaApi, dst: CUdeviceptr, src: *const c_void, size: usize) {
    let result = (cu.cuMemcpyHtoD_v2)(dst, src, size);
    if result != CUDA_SUCCESS {
        panic!("cuMemcpyHtoD_v2 failed: {}", result);
    }
}

/// Save pixel data as a PPM image file (no external crate needed).
fn save_ppm(path: &str, width: u32, height: u32, pixels: &[u32]) {
    use std::io::Write;
    let mut file = std::fs::File::create(path).expect("Failed to create output file");
    write!(file, "P6\n{} {}\n255\n", width, height).unwrap();

    // Pixels are packed as ABGR (from packColor in CUDA)
    // Flip vertically since OptiX renders bottom-up
    for y in (0..height).rev() {
        for x in 0..width {
            let pixel = pixels[(y * width + x) as usize];
            let r = (pixel & 0xFF) as u8;
            let g = ((pixel >> 8) & 0xFF) as u8;
            let b = ((pixel >> 16) & 0xFF) as u8;
            file.write_all(&[r, g, b]).unwrap();
        }
    }
}
