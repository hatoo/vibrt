use optix::*;
use optix::accel::{self, AccelBuildOptions, BuildInput, TriangleArrayInput};
use optix_sys::cuda::{CudaApi, CUDA_SUCCESS};
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
    ($cuda:expr, $call:expr) => {
        assert_eq!($call, CUDA_SUCCESS, "CUDA error");
    };
}

fn main() {
    let cu = CudaApi::load().expect("Failed to load CUDA");
    let optix_handle = optix::init().expect("Failed to initialize OptiX");

    unsafe {
        // --- CUDA init ---
        cuda_check!(cu, (cu.cuInit)(0));
        let mut cu_device: i32 = 0;
        cuda_check!(cu, (cu.cuDeviceGet)(&mut cu_device, 0));
        let mut cu_ctx: CUcontext = ptr::null_mut();
        cuda_check!(cu, (cu.cuCtxCreate_v2)(&mut cu_ctx, 0, cu_device));
        let mut stream: CUstream = ptr::null_mut();
        cuda_check!(cu, (cu.cuStreamCreate)(&mut stream, 0));

        // --- OptiX context ---
        let ctx = DeviceContext::new(&optix_handle, cu_ctx, &DeviceContextOptions::default())
            .expect("Failed to create OptiX context");

        // --- Module ---
        let pipeline_options = PipelineCompileOptions::new("params")
            .traversable_graph_flags(TraversableGraphFlags::ALLOW_SINGLE_GAS)
            .num_payload_values(3)
            .num_attribute_values(3);

        let module_options = ModuleCompileOptions::default();
        let ptx_path = std::path::Path::new(file!()).parent().unwrap().join("devicecode.ptx");
        let ptx = std::fs::read_to_string(&ptx_path)
            .unwrap_or_else(|_| panic!(
                "Failed to load {}. Compile it first:\n  \
                 nvcc -ptx examples/devicecode.cu -o examples/devicecode.ptx \
                 -I\"<OptiX SDK>/include\" -Iexamples --use_fast_math",
                ptx_path.display()
            ));
        let module = Module::new(&ctx, &module_options, &pipeline_options, ptx.as_bytes())
            .expect("Failed to create module")
            .value;

        // --- Program groups ---
        let raygen_pg = ProgramGroup::raygen(&ctx, &module, "__raygen__rg")
            .expect("raygen").value;
        let miss_pg = ProgramGroup::miss(&ctx, &module, "__miss__ms")
            .expect("miss").value;
        let hitgroup_pg = ProgramGroup::hitgroup(&ctx)
            .closest_hit(&module, "__closesthit__ch")
            .build()
            .expect("hitgroup").value;

        // --- Pipeline ---
        let link_options = PipelineLinkOptions { max_trace_depth: 1 };
        let pipeline = Pipeline::new(
            &ctx,
            &pipeline_options,
            &link_options,
            &[&raygen_pg, &miss_pg, &hitgroup_pg],
        )
        .expect("pipeline")
        .value;
        pipeline.set_stack_size(2048, 2048, 2048, 1).expect("stack size");

        // --- Acceleration structure ---
        let vertices: [[f32; 3]; 3] = [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
        ];
        let mut d_vertices: CUdeviceptr = 0;
        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_vertices, mem::size_of_val(&vertices)));
        cuda_check!(cu, (cu.cuMemcpyHtoD_v2)(d_vertices, vertices.as_ptr() as *const c_void, mem::size_of_val(&vertices)));

        let vertex_buffers = [d_vertices];
        let geo_flags = [GeometryFlags::NONE];
        let tri_input = TriangleArrayInput::new(
            &vertex_buffers, 3, VertexFormat::Float3,
            3 * mem::size_of::<f32>() as u32, &geo_flags,
        );

        let build_options = AccelBuildOptions {
            build_flags: BuildFlags::ALLOW_COMPACTION,
            operation: BuildOperation::Build,
        };

        let sizes = accel::accel_compute_memory_usage(&ctx, &build_options, &[BuildInput::Triangles(tri_input)])
            .expect("accel memory");

        let mut d_temp: CUdeviceptr = 0;
        let mut d_output: CUdeviceptr = 0;
        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_temp, sizes.temp_size));
        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_output, sizes.output_size));

        // Need to recreate tri_input since it was moved
        let tri_input2 = TriangleArrayInput::new(
            &vertex_buffers, 3, VertexFormat::Float3,
            3 * mem::size_of::<f32>() as u32, &geo_flags,
        );
        let gas_handle = accel::accel_build(
            &ctx, stream, &build_options, &[BuildInput::Triangles(tri_input2)],
            d_temp, sizes.temp_size, d_output, sizes.output_size,
        ).expect("accel build");

        cuda_check!(cu, (cu.cuStreamSynchronize)(stream));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_temp));

        // --- SBT ---
        let raygen_record = SbtRecord::new(&raygen_pg, RayGenData {}).expect("raygen sbt");
        let miss_record = SbtRecord::new(&miss_pg, MissData { bg_color: [0.1, 0.1, 0.3] }).expect("miss sbt");
        let hitgroup_record = SbtRecord::new(&hitgroup_pg, HitGroupData {}).expect("hitgroup sbt");

        let mut d_rg: CUdeviceptr = 0;
        let mut d_ms: CUdeviceptr = 0;
        let mut d_hg: CUdeviceptr = 0;
        let rg_size = mem::size_of_val(&raygen_record);
        let ms_size = mem::size_of_val(&miss_record);
        let hg_size = mem::size_of_val(&hitgroup_record);

        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_rg, rg_size));
        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_ms, ms_size));
        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_hg, hg_size));
        cuda_check!(cu, (cu.cuMemcpyHtoD_v2)(d_rg, &raygen_record as *const _ as *const c_void, rg_size));
        cuda_check!(cu, (cu.cuMemcpyHtoD_v2)(d_ms, &miss_record as *const _ as *const c_void, ms_size));
        cuda_check!(cu, (cu.cuMemcpyHtoD_v2)(d_hg, &hitgroup_record as *const _ as *const c_void, hg_size));

        let sbt = ShaderBindingTableBuilder::new(d_rg)
            .miss_records(d_ms, ms_size as u32, 1)
            .hitgroup_records(d_hg, hg_size as u32, 1)
            .build()
            .expect("SBT build");

        // --- Output image ---
        let image_size = (WIDTH * HEIGHT) as usize * mem::size_of::<u32>();
        let mut d_image: CUdeviceptr = 0;
        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_image, image_size));

        let params = Params {
            image: d_image,
            image_width: WIDTH,
            image_height: HEIGHT,
            cam_eye: [0.0, 0.0, 2.0],
            cam_u: [1.2, 0.0, 0.0],
            cam_v: [0.0, 0.9, 0.0],
            cam_w: [0.0, 0.0, -1.0],
            handle: gas_handle,
        };

        let mut d_params: CUdeviceptr = 0;
        let params_size = mem::size_of::<Params>();
        cuda_check!(cu, (cu.cuMemAlloc_v2)(&mut d_params, params_size));
        cuda_check!(cu, (cu.cuMemcpyHtoD_v2)(d_params, &params as *const _ as *const c_void, params_size));

        // --- Launch ---
        println!("Launching OptiX render ({} x {})...", WIDTH, HEIGHT);
        pipeline.launch(stream, d_params, params_size, &sbt, WIDTH, HEIGHT, 1)
            .expect("launch");
        cuda_check!(cu, (cu.cuStreamSynchronize)(stream));

        // --- Download and save ---
        let mut pixels = vec![0u32; (WIDTH * HEIGHT) as usize];
        cuda_check!(cu, (cu.cuMemcpyDtoH_v2)(pixels.as_mut_ptr() as *mut c_void, d_image, image_size));
        save_ppm("output.ppm", WIDTH, HEIGHT, &pixels);
        println!("Saved output.ppm ({} x {})", WIDTH, HEIGHT);

        // --- Cleanup ---
        // Drop OptiX resources before destroying CUDA context.
        drop(pipeline);
        drop(hitgroup_pg);
        drop(miss_pg);
        drop(raygen_pg);
        drop(module);
        drop(ctx);

        cuda_check!(cu, (cu.cuMemFree_v2)(d_image));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_params));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_rg));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_ms));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_hg));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_output));
        cuda_check!(cu, (cu.cuMemFree_v2)(d_vertices));

        cuda_check!(cu, (cu.cuStreamDestroy_v2)(stream));
        cuda_check!(cu, (cu.cuCtxDestroy_v2)(cu_ctx));
    }
}

fn save_ppm(path: &str, width: u32, height: u32, pixels: &[u32]) {
    use std::io::Write;
    let mut file = std::fs::File::create(path).expect("Failed to create output file");
    write!(file, "P6\n{} {}\n255\n", width, height).unwrap();
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
