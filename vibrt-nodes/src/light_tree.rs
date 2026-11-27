//! Compact spatial hierarchy for Cycles' light-tree importance bounds.
//! Renderer uploads these nodes unchanged; leaf indices support matching hit PDFs.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub lower: [f32; 3],
    pub energy: f32,
    pub upper: [f32; 3],
    pub theta_o: f32,
    pub axis: [f32; 3],
    pub theta_e: f32,
    pub left: i32,
    pub right: i32,
    pub parent: i32,
    pub emitter: i32,
}

impl Node {
    pub fn emitter(
        lower: [f32; 3],
        upper: [f32; 3],
        energy: f32,
        axis: [f32; 3],
        two_sided: bool,
    ) -> Self {
        Self {
            lower,
            upper,
            energy: energy.max(0.0),
            axis,
            theta_o: if two_sided { std::f32::consts::PI } else { 0.0 },
            theta_e: std::f32::consts::FRAC_PI_2,
            left: -1,
            right: -1,
            parent: -1,
            emitter: -1,
        }
    }
}

#[derive(Default)]
pub struct Tree {
    pub nodes: Vec<Node>,
    pub leaves: Vec<i32>,
}

pub fn build(emitters: &[Node]) -> Tree {
    fn branch(tree: &mut Tree, source: &[Node], indices: &mut [usize], parent: i32) -> i32 {
        let id = tree.nodes.len() as i32;
        let mut node = source[indices[0]];
        node.parent = parent;
        if indices.len() == 1 {
            node.emitter = indices[0] as i32;
            tree.leaves[indices[0]] = id;
            tree.nodes.push(node);
            return id;
        }
        node.energy = 0.0;
        node.theta_o = std::f32::consts::PI;
        node.axis = [0.0, 0.0, 1.0];
        for &i in indices.iter() {
            node.energy += source[i].energy;
            for c in 0..3 {
                node.lower[c] = node.lower[c].min(source[i].lower[c]);
                node.upper[c] = node.upper[c].max(source[i].upper[c]);
            }
        }
        let axis = (0..3)
            .max_by(|&a, &b| {
                (node.upper[a] - node.lower[a]).total_cmp(&(node.upper[b] - node.lower[b]))
            })
            .unwrap();
        indices.sort_unstable_by(|&a, &b| {
            (source[a].lower[axis] + source[a].upper[axis])
                .total_cmp(&(source[b].lower[axis] + source[b].upper[axis]))
        });
        tree.nodes.push(node);
        let mid = indices.len() / 2;
        let (left, right) = indices.split_at_mut(mid);
        tree.nodes[id as usize].left = branch(tree, source, left, id);
        tree.nodes[id as usize].right = branch(tree, source, right, id);
        id
    }
    let mut tree = Tree {
        nodes: Vec::with_capacity(emitters.len().saturating_mul(2)),
        leaves: vec![-1; emitters.len()],
    };
    if !emitters.is_empty() {
        branch(
            &mut tree,
            emitters,
            &mut (0..emitters.len()).collect::<Vec<_>>(),
            -1,
        );
    }
    tree
}
