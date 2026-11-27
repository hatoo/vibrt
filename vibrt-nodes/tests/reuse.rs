use serde_json::json;
use vibrt_nodes::{
    ies,
    light_tree::{self, Node},
    shader::{evaluate, ShaderGraph},
};
unsafe extern "C" {
    fn vibrt_reuse_checks() -> i32;
    fn vibrt_ies_eval(data: *const f32, phi: f32, theta: f32) -> f32;
    fn vibrt_tree_sample(
        nodes: *const Node,
        p: *const f32,
        n: *const f32,
        u: f32,
        pdf: *mut f32,
    ) -> i32;
    fn vibrt_tree_pdf(nodes: *const Node, leaf: i32, p: *const f32, n: *const f32) -> f32;
}

#[test]
fn imported_sampling_kernels_have_matched_densities() {
    assert_eq!(
        unsafe { vibrt_reuse_checks() },
        0,
        "failed line in native/reuse_bridge.cpp"
    );
}

#[test]
fn light_tree_is_normalized_and_sample_matches_hit_probability() {
    assert_eq!(std::mem::size_of::<Node>(), 64);
    assert_eq!(std::mem::offset_of!(Node, emitter), 60);
    assert!(light_tree::build(&[]).nodes.is_empty());
    let emitters: Vec<_> = (0..37)
        .map(|i| {
            let x = i as f32;
            Node::emitter(
                [x, -1.0, 2.0],
                [x + 0.5, 1.0, 2.0],
                if i == 4 { 0.0 } else { x + 1.0 },
                [0.0, 0.0, -1.0],
                false,
            )
        })
        .collect();
    let tree = light_tree::build(&emitters);
    assert_eq!(tree.nodes.len(), 73);
    for p in [[0.0, 0.0, 0.0], [17.0, 0.0, 2.0], [5.0, 4.0, 1000.0]] {
        for n in [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]] {
            let pdfs: Vec<_> = tree
                .leaves
                .iter()
                .map(|&l| unsafe { vibrt_tree_pdf(tree.nodes.as_ptr(), l, p.as_ptr(), n.as_ptr()) })
                .collect();
            assert!((pdfs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert_eq!(pdfs[4], 0.0);
            let mut counts = vec![0; 37];
            for i in 0..8192 {
                let mut pdf = 0.0;
                let selected = unsafe {
                    vibrt_tree_sample(
                        tree.nodes.as_ptr(),
                        p.as_ptr(),
                        n.as_ptr(),
                        (i as f32 + 0.5) / 8192.0,
                        &mut pdf,
                    )
                };
                assert!(selected >= 0);
                let k = selected as usize;
                assert!((pdf - pdfs[k]).abs() < 1e-6);
                counts[k] += 1;
            }
            for (count, pdf) in counts.iter().zip(pdfs) {
                assert!((*count as f32 / 8192.0 - pdf).abs() < 0.0003);
            }
        }
    }
}

fn ies_text(kind: u32, vertical: &str, horizontal: &str, nv: usize, nh: usize) -> String {
    format!("IESNA:LM-63-2002\nTILT=NONE\n1 1000 2 {nv} {nh} {kind} 1 0 0 0\n0.5 1 10\n{vertical}\n{horizontal}\n{}\n",vec!["10";nv*nh].join(" "))
}

#[test]
fn ies_types_symmetry_units_and_interpolation() {
    for (kind, v, h, nv, nh) in [
        (1, "0 90 180", "0", 3, 1),
        (1, "0 90 180", "0 90", 3, 2),
        (2, "-90 0 90", "-90 0 90", 3, 3),
        (3, "-90 0 90", "-90 0 90", 3, 3),
    ] {
        let profile = ies::parse(&ies_text(kind, v, h, nv, nh)).unwrap();
        assert!(profile.cycles_processed);
        assert!((profile.peak_absolute_candela - 0.706_650_8).abs() < 1e-6);
        let data = ies::pack(
            &profile.thetas_deg,
            &profile.phis_deg,
            &profile.candelas,
            profile.peak_candela,
            true,
        )
        .unwrap();
        for phi in [0.1, 1.1, 3.1, 5.1] {
            for theta in [0.2, 0.7, 1.2, 2.5] {
                let value = unsafe { vibrt_ies_eval(data.as_ptr(), phi, theta) };
                assert!(
                    value.is_finite() && (0.0..=1.001).contains(&value),
                    "{kind}: {phi} {theta}: {value}"
                );
                if kind == 1 {
                    assert!((value - 1.0).abs() < 1e-5);
                }
            }
        }
    }
    let legacy = ies::pack(&[0.0, 90.0, 180.0], &[0.0], &[2.0; 3], 2.0, false).unwrap();
    assert!((unsafe { vibrt_ies_eval(legacy.as_ptr(), 5.0, 1.0) } - 1.0).abs() < 1e-5);
}

#[test]
fn ies_rejects_malformed_input_before_native_allocation() {
    for text in [
        "",
        "TILT=NONE\n1 2 3 -1 999999999 1 1 0 0 0 1 1 1",
        "TILT=INCLUDE\n1 99999999",
        "TILT=NONE\n1 1 1 3 1 1 1 0 0 0 1 1 1 0 0 180 0 1 1 1",
    ] {
        assert!(ies::parse(text).is_err());
    }
    assert!(ies::pack(&[0.0, 180.0], &[0.0], &[1.0, f32::NAN], 1.0, false).is_err());
    assert!(ies::pack(&[0.0, 180.0], &[0.0], &[1.0, 1.0], -1.0, false).is_err());
}

fn node(op: &str, values: &[[f32; 4]], modes: &[i32]) -> [f32; 4] {
    let mut nodes: Vec<_> = values
        .iter()
        .map(|v| json!({"op":"constant","params":v}))
        .collect();
    let mut mode_array = [0; 4];
    mode_array[..modes.len()].copy_from_slice(modes);
    nodes.push(json!({"op":op,"inputs":(0..values.len()).collect::<Vec<_>>(),"modes":mode_array}));
    let graph: ShaderGraph =
        serde_json::from_value(json!({"nodes":nodes,"outputs":{"base_color":values.len()}}))
            .unwrap();
    evaluate(&graph, &[[0.0; 128]]).unwrap()[0][0]
}

#[test]
fn value_nodes_preserve_cycles_edge_conventions() {
    let constants = [[0.4; 4], [1.0; 4], [1.0; 4], [2.0; 4], [5.0; 4], [4.0; 4]];
    assert_eq!(node("map_range", &constants, &[0, 0, 0])[0], 0.0);
    assert_eq!(node("map_range", &constants, &[0, 1, 0])[0], 2.0);
    let vector = [3.0, 4.0, 5.0, 1.0];
    assert_eq!(
        &node(
            "vector_rotate",
            &[vector, [1.0; 4], [0.0; 4], [1.2; 4], [0.0; 4]],
            &[0, 0]
        )[..3],
        &vector[..3]
    );
    let normal = node(
        "mapping",
        &[
            [1.0, 1.0, 0.0, 0.0],
            [5.0; 4],
            [0.0; 4],
            [0.0, 2.0, 1.0, 0.0],
        ],
        &[3],
    );
    assert_eq!(&normal[..3], &[0.0, 1.0, 0.0]);
    let result = node(
        "mapping",
        &[
            [3.0, 4.0, 5.0, 0.0],
            [1.0, 2.0, 3.0, 0.0],
            [0.0; 4],
            [2.0, 2.0, 2.0, 0.0],
        ],
        &[1],
    );
    assert_eq!(&result[..3], &[1.0, 1.0, 1.0]);
}
