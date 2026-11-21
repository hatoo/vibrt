// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0
//! Cycles' welded-vertex curvature attribute, adapted from
//! blender-v5.1-release/intern/cycles/blender/mesh.cpp (attr_create_pointiness).

use std::collections::HashSet;

pub fn compute(positions: &[[f32; 3]], normals: &[[f32; 3]], edges: &[[u32; 2]]) -> Vec<f32> {
    let n = positions.len();
    let sums: Vec<f32> = positions.iter().map(|p| p.iter().sum()).collect();
    let mut sorted: Vec<usize> = (0..n).collect();
    // Tie-break every equal sum, not only coincident points: mixing those
    // rules violates the total-order contract when distinct points share a sum.
    sorted.sort_unstable_by(|&a, &b| sums[a].total_cmp(&sums[b]).then_with(|| b.cmp(&a)));
    let mut original: Vec<usize> = (0..n).collect();
    for (order, &a) in sorted.iter().enumerate() {
        for &b in &sorted[order + 1..] {
            if sums[b] - sums[a] > 3.0 * f32::EPSILON {
                break;
            }
            let distance: f32 = (0..3)
                .map(|k| (positions[b][k] - positions[a][k]).powi(2))
                .sum();
            if distance < f32::EPSILON {
                original[a] = b;
                break;
            }
        }
    }
    for i in 0..n {
        let mut root = original[i];
        while root != original[root] {
            root = original[root];
        }
        original[i] = root;
    }
    let mut normal = vec![[0.0f32; 3]; n];
    for i in 0..n {
        for k in 0..3 {
            normal[original[i]][k] += normals[i][k];
        }
    }
    for v in &mut normal {
        let length = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if length > 0.0 {
            for x in v {
                *x /= length;
            }
        }
    }
    let mut visited = HashSet::new();
    let mut unique_edges = Vec::new();
    let mut accumulated = vec![[0.0f32; 3]; n];
    let mut count = vec![0u32; n];
    for edge in edges {
        let a = original[edge[0] as usize];
        let b = original[edge[1] as usize];
        if !visited.insert((a.min(b), a.max(b))) {
            continue;
        }
        unique_edges.push((a, b));
        let delta = std::array::from_fn::<_, 3, _>(|k| positions[b][k] - positions[a][k]);
        let length = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
        if length > 0.0 {
            for k in 0..3 {
                accumulated[a][k] += delta[k] / length;
                accumulated[b][k] -= delta[k] / length;
            }
        }
        count[a] += 1;
        count[b] += 1;
    }
    let raw: Vec<f32> = (0..n)
        .map(|i| {
            if count[i] == 0 {
                return 0.0;
            }
            let cosine: f32 = (0..3)
                .map(|k| normal[i][k] * accumulated[i][k])
                .sum::<f32>()
                / count[i] as f32;
            cosine.clamp(-1.0, 1.0).acos() / std::f32::consts::PI
        })
        .collect();
    let mut blurred = raw.clone();
    for (a, b) in unique_edges {
        blurred[a] += raw[b];
        blurred[b] += raw[a];
    }
    for i in 0..n {
        blurred[i] /= (count[i] + 1) as f32;
    }
    original.into_iter().map(|i| blurred[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::compute;

    #[test]
    fn flat_patch_and_welded_seam_are_neutral() {
        let positions = [
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
        ];
        let normals = [[0., 0., 1.]; 5];
        let edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 1]];
        for value in compute(&positions, &normals, &edges) {
            assert!((value - 0.5).abs() < 1e-6);
        }
        assert_eq!(compute(&[[0., 0., 0.]], &[[0., 0., 1.]], &[]), [0.0]);
    }

    #[test]
    fn distinct_and_coincident_equal_sum_vertices_sort_consistently() {
        // A comparator that only tie-breaks coincident vertices violates
        // transitivity on this mix (the Barbershop export used to panic).
        let positions: Vec<_> = (0..128)
            .map(|i| {
                let x = (i % 17) as f32;
                [x, -x, 0.0]
            })
            .collect();
        let edges: Vec<_> = (1..128).map(|i| [i - 1, i]).collect();
        let values = compute(&positions, &vec![[0., 0., 1.]; 128], &edges);
        assert!(values.iter().all(|v| (*v - 0.5).abs() < 1e-6));
    }
}
