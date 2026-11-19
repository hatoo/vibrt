use serde_json::json;
use vibrt_nodes::shader::{evaluate, ShaderGraph};

fn graph(value: serde_json::Value) -> ShaderGraph {
    serde_json::from_value(value).unwrap()
}

#[test]
fn rejects_invalid_graphs_before_entering_cpp() {
    let constant = json!({"op":"constant", "params":[0.1,0.2,0.3,1.0]});
    for nodes in [
        vec![],
        vec![constant.clone(); 257],
        vec![json!({"op":"missing"})],
        vec![json!({"op":"context","modes":[32,0,0,0]})],
        vec![json!({"op":"math","inputs":[0,0,0]})],
        vec![
            constant.clone(),
            json!({"op":"separate","inputs":[0],"modes":[0,4,0,0]}),
        ],
        vec![
            constant.clone(),
            json!({"op":"ramp","inputs":[0],"table":vec![0.0;4]}),
        ],
    ] {
        let g = graph(json!({"nodes":nodes,"outputs":{"base_color":0}}));
        assert!(evaluate(&g, &[[0.0; 128]]).is_err());
    }
    let g = graph(json!({"nodes":[constant],"outputs":{"base_color":1}}));
    assert!(g.validate().is_err());
}

#[test]
fn reads_dynamic_operands_and_all_output_sockets() {
    let g = graph(json!({"nodes":[
        {"op":"context","modes":[0,0,0,0]},
        {"op":"context","modes":[1,0,0,0]},
        {"op":"constant","params":[0.5,0.5,0.5,1.0]},
        {"op":"math","inputs":[0,1,2],"modes":[36,0,0,0]},
        {"op":"math","inputs":[0,1,2],"modes":[38,0,0,0]}
    ],"outputs":{"roughness":3,"emission_strength":4}}));
    let mut contexts = [[0.0; 128]; 2];
    contexts[0][0] = 0.3;
    contexts[0][4] = 0.7;
    contexts[1][0] = 0.9;
    contexts[1][4] = 0.2;
    let results = evaluate(&g, &contexts).unwrap();
    assert!((results[0][1][0] - 0.71).abs() < 1e-6);
    assert!((results[1][1][0] - 0.68).abs() < 1e-6);
    assert!(results[0][6][0] < 0.3); // Actual smooth minimum, not plain min.
    assert_eq!(results[1][6][0], 0.2);
}

#[test]
fn accepts_boundary_graph_and_preserves_hdr_values() {
    let g = graph(
        json!({"nodes":vec![json!({"op":"constant","params":[-2.0,3.0,4.0,0.25]});256],"outputs":{"base_color":255}}),
    );
    assert_eq!(
        evaluate(&g, &[[0.0; 128]]).unwrap()[0][0],
        [-2.0, 3.0, 4.0, 0.25]
    );
}
