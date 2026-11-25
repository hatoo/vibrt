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
