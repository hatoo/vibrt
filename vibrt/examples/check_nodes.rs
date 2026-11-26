//! Compare the shared node evaluator on CPU and CUDA, including graph packing.
//! cargo run --release -p vibrt --example check_nodes
use anyhow::{ensure, Result};
use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};
use serde_json::json;
use vibrt_native::CudaResultExt;
use vibrt_nodes::{color, graph};

fn main() -> Result<()> {
    let source = format!(
        "{}\n{}\n{}",
        vibrt_nodes::DEVICE_TYPES,
        vibrt_nodes::DEVICE_CORE.replace(
            "#include \"cycles_nodes.h\"",
            &vibrt_nodes::DEVICE_CYCLES.replace(
                "#include \"cycles_compat.h\"",
                vibrt_nodes::DEVICE_CYCLES_COMPAT
            )
        ),
        r#"
extern "C" __global__ void check_nodes(const ColorGraphNode *nodes, int n,
    const float *inputs, float *out, int count) {
  int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  const float *v = inputs + i * 3;
  auto color = vibrt_nodes::eval_color_graph(nodes, n, n-1, {0, 0}, {v[0], v[1], v[2]}, 0);
  out[i*3] = color.x; out[i*3+1] = color.y; out[i*3+2] = color.z;
}
"#
    );
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        &source,
        cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let ctx = CudaContext::new(0).cuda()?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(ptx).cuda()?;
    let kernel = module.load_function("check_nodes").cuda()?;
    let inputs: Vec<f32> = (0..12288).map(|i| (i % 53) as f32 / 19.0 - 0.3).collect();
    let input = stream.clone_htod(&inputs).cuda()?;
    let mut output = stream.alloc_zeros::<f32>(inputs.len()).cuda()?;
    let count = (inputs.len() / 3) as i32;
    let mut cases = Vec::new();
    for blend in [
        "mix",
        "multiply",
        "add",
        "subtract",
        "screen",
        "divide",
        "difference",
        "darken",
        "lighten",
        "overlay",
        "soft_light",
        "linear_light",
        "hue",
        "saturation",
        "color",
        "value",
    ] {
        cases.push(json!({"type": "mix", "blend": blend, "fac": 0.37, "clamp": false}));
    }
    for op in [
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "multiply_add",
        "minimum",
        "maximum",
    ] {
        for swap in [false, true] {
            cases.push(json!({"type": "math", "op": op, "b": 0.7, "c": 0.13, "swap": swap}));
        }
    }
    cases.push(json!({"type": "invert", "fac": 0.61}));
    cases.push(
        json!({"type": "hue_sat", "hue": 0.71, "saturation": 0.82, "value": 1.4, "fac": 0.63}),
    );
    cases.push(json!({"type": "bright_contrast", "bright": 0.12, "contrast": 0.53}));
    let lut: Vec<f32> = (0..768)
        .map(|i| ((i % 256) as f32 / 255.0).powf(1.3))
        .collect();
    cases.push(json!({"type": "rgb_curve", "lut": lut}));
    cases.push(json!({"type": "color_ramp", "lut": [0.1, 0.3, 0.9, 0.9, 0.4, 0.1]}));
    let other = [0.17, 0.42, 0.76];
    for settings in &cases {
        let op: color::ColorOp = serde_json::from_value(settings.clone())?;
        let expected = color::apply(&op, &inputs, Some(&other), None)?;
        let mut node = settings.clone();
        if node["type"] == "mix" {
            node["a"] = json!(0);
            node["b"] = json!(1);
        } else {
            node["input"] = json!(0);
        }
        let graph: graph::ColorGraph = serde_json::from_value(json!({"nodes": [
            {"type": "vertex_color"}, {"type": "const", "rgb": other}, node
        ]}))?;
        let mut luts = Vec::new();
        let mut pointers = vec![0; 3];
        let mut lengths = vec![0; 3];
        if let Some(lut) = node.get("lut") {
            let lut: Vec<f32> = serde_json::from_value(lut.clone())?;
            let buffer = stream.clone_htod(&lut).cuda()?;
            pointers[2] = buffer.device_ptr(&stream).0;
            lengths[2] = (lut.len() / 3) as u32;
            luts.push(buffer);
        }
        let packed = graph::pack(&graph, &[], &pointers, &lengths)?;
        // ColorGraphNode's compile-time assertions guarantee 16 packed u32s.
        let words =
            unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u32>(), packed.len() * 16) };
        let nodes = stream.clone_htod(words).cuda()?;
        let n = packed.len() as i32;
        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&nodes)
                .arg(&n)
                .arg(&input)
                .arg(&mut output)
                .arg(&count)
                .launch(LaunchConfig::for_num_elems(count as u32))
        }
        .cuda()?;
        let actual = stream.clone_dtoh(&output).cuda()?;
        for (a, b) in actual.iter().zip(expected) {
            ensure!(
                (a - b).abs() <= 2e-5 * b.abs().max(1.0),
                "CPU/CUDA difference for {}: {a} vs {b}",
                settings["type"]
            );
        }
    }
    println!(
        "PASS: {} node cases, 4096 RGB inputs each, CPU/CUDA parity",
        cases.len()
    );
    Ok(())
}
