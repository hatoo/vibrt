#pragma once

/// Single node in a material's colour graph. Layout mirrors
/// `gpu_types::ColorGraphNode` on the Rust side.
struct ColorGraphNode {
  unsigned int tag;
  unsigned int payload[15];
};

#define COLOR_NODE_CONST 0u
#define COLOR_NODE_IMAGE_TEX 1u
#define COLOR_NODE_MIX 2u
#define COLOR_NODE_INVERT 3u
#define COLOR_NODE_MATH 4u
#define COLOR_NODE_HUE_SAT 5u
#define COLOR_NODE_RGB_CURVE 6u
#define COLOR_NODE_BRIGHT_CONTRAST 7u
#define COLOR_NODE_VERTEX_COLOR 8u
#define COLOR_NODE_OBJECT_RANDOM 9u
#define COLOR_NODE_COLOR_RAMP 10u

static_assert(sizeof(ColorGraphNode) == 64, "ColorGraphNode ABI");
