//! Compact BVHs for AO, Bevel and Raycast. The path tracer keeps its OptiX BVH.
use crate::gpu_types::QueryNode;
pub type Bounds = ([f32; 3], [f32; 3]);

pub fn build_triangles(vertices: &[f32], indices: &[u32]) -> (Vec<QueryNode>, Vec<u32>) {
    let bounds: Vec<_> = indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|triangle| {
            let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for &index in triangle {
                for axis in 0..3 {
                    let v = vertices[index as usize * 3 + axis];
                    lo[axis] = lo[axis].min(v);
                    hi[axis] = hi[axis].max(v);
                }
            }
            (lo, hi)
        })
        .collect();
    build(&bounds)
}

pub fn build(bounds: &[Bounds]) -> (Vec<QueryNode>, Vec<u32>) {
    if bounds.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut indices: Vec<u32> = (0..bounds.len() as u32).collect();
    let mut nodes = vec![QueryNode::default(); node_count(bounds.len())];
    let (root, descendants) = nodes.split_at_mut(1);
    split(&mut root[0], descendants, &mut indices, bounds, 1, 0);
    (nodes, indices)
}

// Median splits depend only on primitive count. At the last complete level,
// each of `width` nodes has 4..=7 primitives; only those with >4 split again.
// A full binary tree with L leaves has 2*L-1 nodes.
fn node_count(primitives: usize) -> usize {
    if primitives <= 4 {
        return usize::from(primitives != 0);
    }
    let width = 1usize << (primitives / 4).ilog2();
    let leaves = width + (primitives - 4 * width).min(width);
    2 * leaves - 1
}

fn split(
    node: &mut QueryNode,
    descendants: &mut [QueryNode],
    indices: &mut [u32],
    bounds: &[Bounds],
    first: usize,
    start: usize,
) {
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for &id in indices.iter() {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(bounds[id as usize].0[axis]);
            hi[axis] = hi[axis].max(bounds[id as usize].1[axis]);
        }
    }
    *node = QueryNode {
        lo,
        hi,
        first: start as u32,
        count: indices.len() as u32,
    };
    if indices.len() <= 4 {
        return;
    }
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
        .unwrap();
    let middle = indices.len() / 2;
    indices.select_nth_unstable_by(middle, |&a, &b| {
        let a = bounds[a as usize];
        let b = bounds[b as usize];
        (a.0[axis] + a.1[axis]).total_cmp(&(b.0[axis] + b.1[axis]))
    });
    node.first = first as u32;
    node.count = 0;

    // Keep the serial builder's sibling-pair layout: two children, then all
    // left descendants, then all right descendants. Disjoint slices let Rayon
    // fill both subtrees without relocation, locks or extra temporary trees.
    let parallel = indices.len() >= 16_384;
    let (children, rest) = descendants.split_at_mut(2);
    let (left_node, right_node) = children.split_at_mut(1);
    let (left_nodes, right_nodes) = rest.split_at_mut(node_count(middle) - 1);
    let right_first = first + 2 + left_nodes.len();
    let (left, right) = indices.split_at_mut(middle);
    let mut build_left = || {
        split(
            &mut left_node[0],
            left_nodes,
            left,
            bounds,
            first + 2,
            start,
        )
    };
    let mut build_right = || {
        split(
            &mut right_node[0],
            right_nodes,
            right,
            bounds,
            right_first,
            start + middle,
        )
    };
    if parallel {
        rayon::join(build_left, build_right);
    } else {
        build_left();
        build_right();
    }
}

#[cfg(test)]
fn build_serial_reference(bounds: &[Bounds]) -> (Vec<QueryNode>, Vec<u32>) {
    if bounds.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut indices: Vec<u32> = (0..bounds.len() as u32).collect();
    let mut nodes = vec![QueryNode::default()];
    fn split(
        nodes: &mut Vec<QueryNode>,
        indices: &mut [u32],
        bounds: &[Bounds],
        node: usize,
        start: usize,
    ) {
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for &id in indices.iter() {
            for axis in 0..3 {
                lo[axis] = lo[axis].min(bounds[id as usize].0[axis]);
                hi[axis] = hi[axis].max(bounds[id as usize].1[axis]);
            }
        }
        nodes[node] = QueryNode {
            lo,
            hi,
            first: start as u32,
            count: indices.len() as u32,
        };
        if indices.len() <= 4 {
            return;
        }
        let axis = (0..3)
            .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
            .unwrap();
        let middle = indices.len() / 2;
        indices.select_nth_unstable_by(middle, |&a, &b| {
            let a = bounds[a as usize];
            let b = bounds[b as usize];
            (a.0[axis] + a.1[axis]).total_cmp(&(b.0[axis] + b.1[axis]))
        });
        let first = nodes.len();
        nodes.extend([QueryNode::default(); 2]);
        nodes[node].first = first as u32;
        nodes[node].count = 0;
        let (left, right) = indices.split_at_mut(middle);
        split(nodes, left, bounds, first, start);
        split(nodes, right, bounds, first + 1, start + middle);
    }
    split(&mut nodes, &mut indices, bounds, 0, 0);
    (nodes, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_count_matches_recursive_topology() {
        let mut expected = vec![0; 65_538];
        for n in 1..expected.len() {
            expected[n] = if n <= 4 {
                1
            } else {
                1 + expected[n / 2] + expected[n - n / 2]
            };
            assert_eq!(node_count(n), expected[n], "primitive count {n}");
        }
        assert_eq!(node_count(0), 0);
    }

    #[test]
    fn parallel_builder_matches_serial_layout_and_primitive_order() {
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for n in [0, 1, 4, 5, 9, 17, 8193, 16383, 16384, 16385, 65537] {
                    for uniform in [false, true] {
                        let bounds: Vec<_> = (0..n)
                            .map(|i| {
                                // Include ties, planar/degenerate bounds, and a
                                // shuffled dominant axis at irregular lengths.
                                if uniform {
                                    ([0.0; 3], [0.0; 3])
                                } else {
                                    let x = ((i as u32).wrapping_mul(2654435761) % 65521) as f32;
                                    let lo = [x, (i % 11) as f32, 0.0];
                                    (lo, [x + (i % 3) as f32, lo[1], 0.0])
                                }
                            })
                            .collect();
                        let actual = build(&bounds);
                        let expected = build_serial_reference(&bounds);
                        assert_eq!(actual.1, expected.1);
                        assert_eq!(actual.0.len(), expected.0.len());
                        for (a, b) in actual.0.iter().zip(&expected.0) {
                            assert_eq!(a.lo.map(f32::to_bits), b.lo.map(f32::to_bits));
                            assert_eq!(a.hi.map(f32::to_bits), b.hi.map(f32::to_bits));
                            assert_eq!((a.first, a.count), (b.first, b.count));
                        }
                    }
                }
            });
        }
    }

    #[test]
    fn indexed_triangles_preserve_primitive_order_and_bounds() {
        let vertices = [-2.0, 3.0, 1.0, 4.0, -1.0, 2.0, 0.0, 5.0, -3.0];
        let indices = [2, 0, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 0, 2];
        let bounds = [
            ([-2.0, -1.0, -3.0], [4.0, 5.0, 2.0]),
            ([4.0, -1.0, 2.0], [4.0, -1.0, 2.0]),
            ([-2.0, 3.0, 1.0], [-2.0, 3.0, 1.0]),
            ([0.0, 5.0, -3.0], [0.0, 5.0, -3.0]),
            ([-2.0, -1.0, -3.0], [4.0, 5.0, 2.0]),
        ];
        let actual = build_triangles(&vertices, &indices);
        let expected = build(&bounds);
        assert_eq!(actual.1, expected.1);
        assert_eq!(actual.0.len(), expected.0.len());
        for (a, b) in actual.0.iter().zip(&expected.0) {
            assert_eq!(
                (a.lo, a.hi, a.first, a.count),
                (b.lo, b.hi, b.first, b.count)
            );
        }
        let empty = build_triangles(&[], &[]);
        assert!(empty.0.is_empty() && empty.1.is_empty());
    }

    #[test]
    fn leaves_cover_primitives_once_and_parents_enclose_children() {
        let bounds: Vec<_> = (0..257)
            .map(|i| ([i as f32, -1.0, 0.0], [i as f32 + 0.5, 1.0, 0.0]))
            .collect();
        let (nodes, indices) = build(&bounds);
        let mut seen = Vec::new();
        for n in &nodes {
            if n.count > 0 {
                seen.extend_from_slice(&indices[n.first as usize..(n.first + n.count) as usize]);
            } else {
                for child in &nodes[n.first as usize..n.first as usize + 2] {
                    for a in 0..3 {
                        assert!(n.lo[a] <= child.lo[a] && n.hi[a] >= child.hi[a]);
                    }
                }
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..257).collect::<Vec<u32>>());
        assert!(build(&[]).0.is_empty());
    }
}
