/// Single node in a material's colour graph. The graph is a sequential
/// list; each node reads from earlier nodes by index and writes one RGB
/// value. The device-side evaluator allocates a small stack of `float3`
/// slots (one per node) and runs through them in order.
///
/// Fields are type-punned into `payload[]` (u32 for indices / tags,
/// bit-cast float for constants). Host side fills the packing; device
/// side mirrors the layout via the same struct in `native/types.h`.
///
/// Payload layout per tag (`payload[i]` as u32 unless noted):
///   Const       [0..3] = rgb (float bitcast)
///   ImageTex    [0] = tex_ptr_lo, [1] = tex_ptr_hi, [2] = w, [3] = h,
///               [4] = channels, [5..11] = uv_transform (6 floats)
///   Mix         [0] = in_a, [1] = in_b, [2] = blend_type, [3] = fac_src
///               (0=const, 1=node), [4] = fac_node, [5] = clamp (0/1),
///               [6] = fac_const (float)
///   Invert      [0] = in, [1] = fac (float)
///   Math        [0] = in, [1] = op, [2] = clamp (0/1),
///               [3] = b_const (float), [4] = c_const (float),
///               [5] = swap (0/1) — flips operand order for non-commutative
///                     ops (subtract/divide/power); ignored for others.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ColorGraphNode {
    pub tag: u32,
    pub payload: [u32; 15],
}

// Layout contract with `native/types.h`: 16 × u32, no padding, 4-byte aligned.
// Host-side uploaders rely on this to reinterpret `&[ColorGraphNode]` as `&[u32]`.
const _: () = assert!(std::mem::size_of::<ColorGraphNode>() == 64);
const _: () = assert!(std::mem::align_of::<ColorGraphNode>() == 4);

// Must stay in sync with `native/types.h` and `scene_loader.rs`.
pub const COLOR_NODE_CONST: u32 = 0;
pub const COLOR_NODE_IMAGE_TEX: u32 = 1;
pub const COLOR_NODE_MIX: u32 = 2;
pub const COLOR_NODE_INVERT: u32 = 3;
pub const COLOR_NODE_MATH: u32 = 4;
pub const COLOR_NODE_HUE_SAT: u32 = 5;
pub const COLOR_NODE_RGB_CURVE: u32 = 6;
pub const COLOR_NODE_BRIGHT_CONTRAST: u32 = 7;
pub const COLOR_NODE_VERTEX_COLOR: u32 = 8;
pub const COLOR_NODE_OBJECT_RANDOM: u32 = 9;
pub const COLOR_NODE_COLOR_RAMP: u32 = 10;

/// Texture address, width, height and f32 channel count: L, LA, RGB or RGBA.
/// L replicates into RGB; layouts without alpha reconstruct alpha exactly as 1.
pub type TextureBinding = (u64, i32, i32, i32);
