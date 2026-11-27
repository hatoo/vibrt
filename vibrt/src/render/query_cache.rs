//! Bounded, owning CPU query trees for repeated renders of unchanged geometry.

use crate::{gpu_types::QueryNode, query_bvh, scene_loader::LoadedMesh};
use rayon::prelude::*;
use std::{mem, sync::Arc};

const CACHE_BYTES: usize = 256 * 1024 * 1024;

#[derive(Default)]
pub(super) struct QueryTree {
    pub nodes: Box<[QueryNode]>,
    pub indices: Box<[u32]>,
}

struct Entry {
    vertices: Box<[f32]>,
    radii: Box<[f32]>,
    indices: Box<[u32]>,
    tree: Arc<QueryTree>,
}

impl Entry {
    fn matches(&self, mesh: &LoadedMesh) -> bool {
        // Compare bits, including signed zero and NaN payloads. Neither RNA
        // addresses nor a hash can establish that the geometry is unchanged.
        bytemuck::cast_slice::<f32, u8>(&self.vertices)
            == bytemuck::cast_slice::<f32, u8>(&mesh.vertices)
            && *self.indices == *mesh.indices
            && bytemuck::cast_slice::<f32, u8>(&self.radii)
                == bytemuck::cast_slice::<f32, u8>(&mesh.curve_radii)
    }
}

pub(super) struct QueryCache {
    entries: Vec<Option<Entry>>,
    keep: Vec<bool>,
    budget: usize,
}

impl Default for QueryCache {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            keep: Vec::new(),
            budget: CACHE_BYTES,
        }
    }
}

impl QueryCache {
    /// Retain only slots that fit this render's budget. Slot order is merely a
    /// lookup hint: every hit still requires an exact match of geometry arrays.
    pub(super) fn begin(&mut self, meshes: &[LoadedMesh]) {
        self.entries.resize_with(meshes.len(), || None);
        self.keep.clear();
        let mut remaining = self.budget;
        for (entry, mesh) in self.entries.iter_mut().zip(meshes) {
            let primitives = mesh.indices.len() / 3;
            let bytes = mem::size_of_val(&mesh.vertices[..])
                + mem::size_of_val(&mesh.indices[..])
                + mem::size_of_val(&mesh.curve_radii[..])
                + query_bvh::node_count(primitives) * mem::size_of::<QueryNode>()
                + primitives * mem::size_of::<u32>();
            let keep = bytes <= remaining;
            if keep {
                remaining -= bytes;
            }
            // Drop stale sizes before replacing any slots, so even a scene
            // switch cannot retain two scenes' worth of cached payloads.
            if !keep
                || entry.as_ref().is_some_and(|e| {
                    e.vertices.len() != mesh.vertices.len()
                        || e.indices.len() != mesh.indices.len()
                        || e.radii.len() != mesh.curve_radii.len()
                })
            {
                *entry = None;
            }
            self.keep.push(keep);
        }
    }

    pub(super) fn batch(&mut self, first: usize, meshes: &[LoadedMesh]) -> Vec<Arc<QueryTree>> {
        let end = first + meshes.len();
        self.entries[first..end]
            .par_iter_mut()
            .zip(&self.keep[first..end])
            .zip(meshes)
            .map(|((entry, &keep), mesh)| {
                if let Some(hit) = entry.as_ref().filter(|e| e.matches(mesh)) {
                    return hit.tree.clone();
                }
                // Release changed geometry before allocating its replacement.
                *entry = None;
                let (nodes, indices) =
                    query_bvh::build_geometry(&mesh.vertices, &mesh.indices, &mesh.curve_radii);
                let tree = Arc::new(QueryTree {
                    nodes: nodes.into_boxed_slice(),
                    indices: indices.into_boxed_slice(),
                });
                if keep {
                    *entry = Some(Entry {
                        vertices: mesh.vertices.to_vec().into_boxed_slice(),
                        radii: mesh.curve_radii.to_vec().into_boxed_slice(),
                        indices: mesh.indices.to_vec().into_boxed_slice(),
                        tree: tree.clone(),
                    });
                }
                tree
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    fn mesh(vertices: &[f32], indices: &[u32]) -> LoadedMesh<'static> {
        LoadedMesh {
            vertices: Cow::Owned(vertices.to_vec()),
            indices: Cow::Owned(indices.to_vec()),
            attributes: Vec::new(),
            normals: Cow::Borrowed(&[]),
            uvs: Cow::Borrowed(&[]),
            material_indices: Cow::Borrowed(&[]),
            vertex_colors: Cow::Borrowed(&[]),
            vertex_alphas: Cow::Borrowed(&[]),
            tangents: Cow::Borrowed(&[]),
            curve_radii: Cow::Borrowed(&[]),
        }
    }

    fn triangle(x: f32) -> LoadedMesh<'static> {
        mesh(&[x, 0.0, 0.0, x + 1.0, 0.0, 0.0, x, 1.0, 0.0], &[0, 1, 2])
    }

    fn prepare(cache: &mut QueryCache, meshes: &[LoadedMesh]) -> Vec<Arc<QueryTree>> {
        cache.begin(meshes);
        cache.batch(0, meshes)
    }

    fn assert_current(tree: &QueryTree, mesh: &LoadedMesh) {
        let (nodes, indices) =
            query_bvh::build_geometry(&mesh.vertices, &mesh.indices, &mesh.curve_radii);
        assert_eq!(*tree.indices, indices);
        assert_eq!(tree.nodes.len(), nodes.len());
        for (actual, expected) in tree.nodes.iter().zip(nodes) {
            assert_eq!(actual.lo.map(f32::to_bits), expected.lo.map(f32::to_bits));
            assert_eq!(actual.hi.map(f32::to_bits), expected.hi.map(f32::to_bits));
            assert_eq!(
                (actual.first, actual.count),
                (expected.first, expected.count)
            );
        }
    }

    #[test]
    fn cache_owns_geometry_and_reuses_new_allocations() {
        let mut cache = QueryCache::default();
        let first = prepare(&mut cache, &[triangle(0.0)]);
        // The temporary source has already dropped. New arrays match by value.
        let next = prepare(&mut cache, &[triangle(0.0)]);
        assert!(Arc::ptr_eq(&first[0], &next[0]));
        assert_current(&next[0], &triangle(0.0));
    }

    #[test]
    fn edits_at_same_addresses_and_slot_reordering_invalidate() {
        let mut cache = QueryCache::default();
        let mut meshes = [triangle(0.0), triangle(2.0)];
        let original = prepare(&mut cache, &meshes);
        meshes[0].vertices.to_mut()[0] = -5.0;
        meshes[1].indices.to_mut()[2] = 1;
        let edited = prepare(&mut cache, &meshes);
        for i in 0..2 {
            assert!(!Arc::ptr_eq(&original[i], &edited[i]));
            assert_current(&edited[i], &meshes[i]);
        }
        meshes.swap(0, 1);
        let reordered = prepare(&mut cache, &meshes);
        for i in 0..2 {
            assert!(!Arc::ptr_eq(&edited[i], &reordered[i]));
            assert_current(&reordered[i], &meshes[i]);
        }
    }

    #[test]
    fn curve_radius_edits_rebuild_bounds() {
        let mut cache = QueryCache::default();
        let mut curve = mesh(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], &[0, 1, 0]);
        curve.curve_radii = Cow::Owned(vec![0.1, 0.2]);
        let mut meshes = [curve];
        let first = prepare(&mut cache, &meshes);
        let same = prepare(&mut cache, &meshes);
        assert!(Arc::ptr_eq(&first[0], &same[0]));
        meshes[0].curve_radii.to_mut()[1] = 0.5;
        let edited = prepare(&mut cache, &meshes);
        assert!(!Arc::ptr_eq(&first[0], &edited[0]));
        assert_current(&edited[0], &meshes[0]);
        assert_eq!(edited[0].nodes[0].hi[0], 0.5);
    }

    #[test]
    fn geometry_key_compares_float_bits() {
        let mut cache = QueryCache::default();
        let mut meshes = [triangle(0.0)];
        let positive = prepare(&mut cache, &meshes);
        meshes[0].vertices.to_mut()[0] = -0.0;
        let negative = prepare(&mut cache, &meshes);
        assert!(!Arc::ptr_eq(&positive[0], &negative[0]));
        meshes[0].vertices.to_mut()[0] = f32::from_bits(0x7fc00001);
        let nan = prepare(&mut cache, &meshes);
        assert!(Arc::ptr_eq(&nan[0], &prepare(&mut cache, &meshes)[0]));
        meshes[0].vertices.to_mut()[0] = f32::from_bits(0x7fc00002);
        assert!(!Arc::ptr_eq(&nan[0], &prepare(&mut cache, &meshes)[0]));
    }

    #[test]
    fn budget_resize_and_oversized_meshes_do_not_retain_stale_entries() {
        let mut cache = QueryCache {
            budget: 84, // 36 vertex + 12 index + 32 node + 4 primitive bytes.
            ..Default::default()
        };
        let meshes = [triangle(0.0), triangle(2.0)];
        let first = prepare(&mut cache, &meshes);
        let next = prepare(&mut cache, &meshes);
        assert!(Arc::ptr_eq(&first[0], &next[0]));
        assert!(!Arc::ptr_eq(&first[1], &next[1]));
        assert!(cache.entries[1].is_none());
        let larger = [mesh(&[0.0; 12], &[0, 1, 2, 0, 2, 3]), triangle(2.0)];
        let trees = prepare(&mut cache, &larger);
        assert!(cache.entries[0].is_none());
        assert!(cache.entries[1].is_some());
        for (tree, mesh) in trees.iter().zip(&larger) {
            assert_current(tree, mesh);
        }
        prepare(&mut cache, &[]);
        assert!(cache.entries.is_empty());
        assert!(cache.keep.is_empty());
    }

    #[test]
    fn batched_hits_preserve_scene_order_and_empty_geometry() {
        let mut cache = QueryCache::default();
        let mut meshes: Vec<_> = (0..65).map(|i| triangle(i as f32)).collect();
        meshes[32] = mesh(&[], &[]);
        let first = prepare(&mut cache, &meshes);
        cache.begin(&meshes);
        for (batch, chunk) in meshes.chunks(32).enumerate() {
            for (i, tree) in cache.batch(batch * 32, chunk).iter().enumerate() {
                assert!(Arc::ptr_eq(tree, &first[batch * 32 + i]));
                assert_current(tree, &chunk[i]);
            }
        }
    }
}
