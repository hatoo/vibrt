use serde_json::{json, Value};
use vibrt_nodes::{
    color::{apply, ColorOp},
    fold,
    graph::{self, ColorGraph},
};

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() <= 2e-5 * b.abs().max(1.0), "{a} != {b}");
    }
}

#[test]
fn preserves_pre_refactor_constant_folding() {
    // Captured from the independently implemented Python evaluator before
    // replacing it; covers all 25 existing unittest cases, including skips.
    let fixtures: Vec<Value> = serde_json::from_str(include_str!("fixtures/fold.json")).unwrap();
    assert_eq!(fixtures.len(), 25);
    for case in fixtures {
        let graph: ColorGraph = serde_json::from_value(case["input"].clone()).unwrap();
        let actual = fold::constants(&graph);
        for (index, expected) in case["expected"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            if expected["type"] == "const" {
                let values: [f32; 3] = serde_json::from_value(expected["rgb"].clone()).unwrap();
                close(&actual[index].unwrap(), &values);
            } else {
                assert!(actual[index].is_none());
            }
        }
    }
}

#[test]
fn preserves_scalar_math_constants_and_invalid_domains() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/scalar_math.json")).unwrap();
    assert_eq!(cases.len(), 222);
    for case in cases {
        let op: ColorOp = serde_json::from_value(case["settings"].clone()).unwrap();
        let input: [f32; 3] = serde_json::from_value(case["input"].clone()).unwrap();
        let actual = apply(&op, &input, None, None);
        if case["expected"].is_null() {
            assert!(actual.is_err(), "{}", case["settings"]);
        } else {
            close(
                &actual.unwrap(),
                &[case["expected"].as_f64().unwrap() as f32; 3],
            );
        }
    }
}

#[test]
fn broadcasts_large_batches_without_mutating_inputs() {
    let a = vec![0.2, 0.4, 0.8].repeat(9000);
    let before = a.clone();
    let op = ColorOp::Mix {
        fac: 0.75,
        blend: "mix".into(),
        clamp: false,
    };
    let factors = vec![0.5; 9000];
    let actual = apply(&op, &a, Some(&[0.8, 0.6, 0.2]), Some(&factors)).unwrap();
    for rgb in actual.chunks_exact(3) {
        close(rgb, &[0.5, 0.5, 0.5]);
    }
    assert_eq!(a, before);
}

#[test]
fn hdr_curve_extrapolation_and_gamma() {
    let op = ColorOp::RgbCurve {
        lut: vec![0.0, 0.5, 0.0, 1.0, 0.0, 2.0],
        low: 0.0,
        high: 1.0,
        fac: 0.5,
        extrapolate: true,
    };
    close(
        &apply(&op, &[-1.0, 0.5, 2.0], None, None).unwrap(),
        &[-0.75, 0.5, 3.0],
    );
    let gamma = ColorOp::Gamma {
        gamma: 2.0,
        preserve_negative: true,
    };
    close(
        &apply(&gamma, &[-0.5, 0.0, 2.0], None, None).unwrap(),
        &[-0.5, 0.0, 4.0],
    );
}

#[test]
fn invalid_inputs_stop_before_ffi() {
    let op = ColorOp::Invert { fac: 1.0 };
    assert!(apply(&op, &[], None, None).is_err());
    assert!(apply(&op, &[0.0, 1.0], None, None).is_err());
    assert!(apply(&op, &[f32::NAN, 0.0, 0.0], None, None).is_err());
    assert!(apply(&op, &[0.0; 6], Some(&[0.0; 9]), None).is_err());
    let curve = ColorOp::RgbCurve {
        lut: vec![0.0; 3],
        low: 0.0,
        high: 1.0,
        fac: 1.0,
        extrapolate: false,
    };
    assert!(apply(&curve, &[0.0; 3], None, None).is_err());
}

#[test]
fn graph_rejects_cycles_missing_bindings_and_stack_overflow() {
    let parse = |v| serde_json::from_value::<ColorGraph>(v).unwrap();
    let cycle = parse(json!({"nodes": [{"type": "invert", "input": 0}]}));
    assert!(graph::pack(&cycle, &[], &[0], &[0]).is_err());
    let oversized = parse(json!({"nodes": vec![json!({"type": "const", "rgb": [0, 0, 0]}); 33]}));
    assert!(oversized.validate().is_err());
    let bad_output = parse(json!({"nodes": [{"type": "const", "rgb": [0,0,0]}], "output": 1}));
    assert!(bad_output.validate().is_err());
    let curve = parse(
        json!({"nodes": [{"type": "const", "rgb": [0,0,0]}, {"type": "rgb_curve", "input": 0, "lut": vec![0.0;768]}]}),
    );
    assert!(graph::pack(&curve, &[], &[], &[]).is_err());
}
