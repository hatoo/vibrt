//! Renderer-independent node definitions and C++ evaluators.
//!
//! Rust validates inputs and owns memory. CPU batches and CUDA shaders share
//! the same color implementation; atmosphere models are compiled for the CPU.
//! This crate has no Blender, Python, CUDA-driver or OptiX dependency.

pub mod color;
pub mod fold;
pub mod graph;
pub mod ies;
pub mod light_tree;
pub mod shader;
pub mod sky;
pub mod types;

/// Embed these in the renderer's NVRTC translation unit. A packaged extension
/// does not need a source checkout or filesystem include paths at runtime.
pub const DEVICE_TYPES: &str = include_str!("../native/types.h");
pub const DEVICE_CORE: &str = include_str!("../native/core.h");
pub const DEVICE_SHADER: &str = include_str!("../native/shader.h");
pub const DEVICE_BSDF: &str = include_str!("../native/bsdf.h");
pub const DEVICE_CYCLES_COMPAT: &str = include_str!("../native/cycles_compat.h");
pub const DEVICE_CYCLES: &str = include_str!(concat!(env!("OUT_DIR"), "/cycles_nodes.h"));
