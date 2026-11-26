// Shared color-node implementation for CPU batches and CUDA shaders.
#pragma once
#include "cycles_nodes.h"
#ifndef __CUDACC__
#include <cmath>
#include <cstring>
#endif

namespace vibrt_nodes {
struct float2 { float x, y; };
struct float3 { float x, y, z; };
VN_INLINE float2 make_float2(float x, float y) { return {x, y}; }
VN_INLINE float3 make_float3(float x, float y, float z) { return {x, y, z}; }
VN_INLINE float bits_float(unsigned int bits) {
#ifdef __CUDACC__
  return __uint_as_float(bits);
#else
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
#endif
}
#define COLOR_GRAPH_MAX_NODES 32
VN_INLINE float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
VN_INLINE float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
VN_INLINE float3 operator-(float3 a) {
  return make_float3(-a.x, -a.y, -a.z);
}
VN_INLINE float3 operator*(float3 a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}
VN_INLINE float3 operator*(float s, float3 a) {
  return a * s;
}
VN_INLINE float3 operator*(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
VN_INLINE float3 operator/(float3 a, float s) {
  return a * (1.0f / s);
}
VN_INLINE float luminance(float3 c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

VN_DEVICE float3 sample_rgba(const float *data, int w, int h,
                                     int channels, float2 uv) {
  if (data == nullptr || w <= 0 || h <= 0)
    return make_float3(1, 1, 1);
  float u = uv.x - floorf(uv.x);
  float v = uv.y - floorf(uv.y);
  float fx = u * (float)w - 0.5f;
  // Blender's image.pixels (and what the exporter writes to the bin) is
  // bottom-up: buffer row 0 is the bottom of the image. Blender UV v=0 is
  // also the bottom, so no flip on the y axis.
  float fy = v * (float)h - 0.5f;
  int x0 = (int)floorf(fx);
  int y0 = (int)floorf(fy);
  float dx = fx - (float)x0;
  float dy = fy - (float)y0;
  auto wrap = [](int v, int m) {
    int r = v % m;
    return r < 0 ? r + m : r;
  };
  int x1 = wrap(x0 + 1, w);
  int y1 = wrap(y0 + 1, h);
  x0 = wrap(x0, w);
  y0 = wrap(y0, h);
  auto fetch = [&](int x, int y) {
    const float *p = &data[(y * w + x) * channels];
    return channels <= 2 ? make_float3(p[0], p[0], p[0])
                         : make_float3(p[0], p[1], p[2]);
  };
  float3 c00 = fetch(x0, y0);
  float3 c10 = fetch(x1, y0);
  float3 c01 = fetch(x0, y1);
  float3 c11 = fetch(x1, y1);
  float3 c0 = c00 * (1.0f - dx) + c10 * dx;
  float3 c1 = c01 * (1.0f - dx) + c11 * dx;
  return c0 * (1.0f - dy) + c1 * dy;
}

// The legacy wire format keeps its operation IDs and input conventions.
// Numerical operations are shared with the general Cycles shader evaluator.
VN_INLINE vibrt_cycles::float3 cycles_rgb(float3 c) { return vibrt_cycles::make_float3(c.x, c.y, c.z); }
VN_INLINE float3 legacy_rgb(vibrt_cycles::float3 c) { return {c.x, c.y, c.z}; }
VN_INLINE float3 rgb_to_hsv_bl(float3 c) {
  auto r = vibrt_cycles::rgb_to_hsv(cycles_rgb(c));
  if (r.z <= 0) r.x = r.y = 0; // historical negative RGB convention
  return legacy_rgb(r);
}
VN_INLINE float3 hsv_to_rgb_bl(float3 c) {
  c.x -= floorf(c.x);
  c.y = fminf(fmaxf(c.y, 0), 1);
  return legacy_rgb(vibrt_cycles::hsv_to_rgb(cycles_rgb(c)));
}
VN_INLINE float3 mix_blend_rgb(float3 a, float3 b, float fac, unsigned int blend, bool clamp_out) {
  using namespace vibrt_cycles;
  const NodeMix modes[] = {NODE_MIX_BLEND, NODE_MIX_MUL, NODE_MIX_ADD,
    NODE_MIX_SUB, NODE_MIX_SCREEN, NODE_MIX_DIV, NODE_MIX_DIFF, NODE_MIX_DARK,
    NODE_MIX_LIGHT, NODE_MIX_OVERLAY, NODE_MIX_SOFT, NODE_MIX_LINEAR,
    NODE_MIX_HUE, NODE_MIX_SAT, NODE_MIX_COL, NODE_MIX_VAL};
  auto r = svm_mix(modes[blend < 16 ? blend : 0], fac, cycles_rgb(a), cycles_rgb(b));
  return legacy_rgb(clamp_out ? clamp(r, 0, 1) : r);
}
VN_INLINE float scalar_math(float a, unsigned int op, float b, float c) {
  using namespace vibrt_cycles;
  const NodeMathType modes[] = {NODE_MATH_ADD, NODE_MATH_SUBTRACT, NODE_MATH_MULTIPLY,
    NODE_MATH_DIVIDE, NODE_MATH_POWER, NODE_MATH_MULTIPLY_ADD, NODE_MATH_MINIMUM,
    NODE_MATH_MAXIMUM, NODE_MATH_LOGARITHM, NODE_MATH_LESS_THAN, NODE_MATH_GREATER_THAN,
    NODE_MATH_COMPARE, NODE_MATH_MODULO, NODE_MATH_MINIMUM, NODE_MATH_MAXIMUM,
    NODE_MATH_ARCTAN2, NODE_MATH_SQRT, NODE_MATH_INV_SQRT, NODE_MATH_ABSOLUTE,
    NODE_MATH_EXPONENT, NODE_MATH_SIGN, NODE_MATH_ROUND, NODE_MATH_FLOOR,
    NODE_MATH_CEIL, NODE_MATH_FRACTION, NODE_MATH_TRUNC, NODE_MATH_SINE,
    NODE_MATH_COSINE, NODE_MATH_TANGENT, NODE_MATH_ARCSINE, NODE_MATH_ARCCOSINE,
    NODE_MATH_ARCTANGENT, NODE_MATH_SINH, NODE_MATH_COSH, NODE_MATH_TANH,
    NODE_MATH_RADIANS, NODE_MATH_DEGREES};
  // These legacy folding edge cases are part of the saved color-graph API.
  if (op == 8 && !(a > 0 && b > 0 && b != 1)) return nanf("");
  if (op == 21) return rintf(a);
  if (op == 29 || op == 30) a = fmaxf(-1, fminf(1, a));
  return svm_math(modes[op < 37 ? op : 0], a, b, c);
}
VN_INLINE float3 math_apply_rgb(float3 inp, unsigned int op, float b, float c, bool swap) {
  auto channel = [&](float a) {
    float x = swap && (op == 1 || op == 3 || op == 4) ? b : a;
    float y = swap && (op == 1 || op == 3 || op == 4) ? a : b;
    if (op == 4) x = fmaxf(x, 0); // legacy image Math clamps the base
    return scalar_math(x, op, y, c);
  };
  return {channel(inp.x), channel(inp.y), channel(inp.z)};
}

VN_INLINE float3 invert_rgb(float3 rgb, float fac) {
  float inv = 1.0f - fac;
  return {rgb.x * inv + (1.0f - rgb.x) * fac,
          rgb.y * inv + (1.0f - rgb.y) * fac,
          rgb.z * inv + (1.0f - rgb.z) * fac};
}
VN_INLINE float3 hue_sat_rgb(float3 src, float hue, float sat, float val, float fac, bool positive) {
  float3 hsv = rgb_to_hsv_bl(src);
  hsv.x = hsv.x + (hue - 0.5f);
  hsv.y = fmaxf(0.0f, fminf(1.0f, hsv.y * sat));
  hsv.z *= val;
  float3 shifted = hsv_to_rgb_bl(hsv);
  if (positive) shifted = {fmaxf(0.0f, shifted.x), fmaxf(0.0f, shifted.y), fmaxf(0.0f, shifted.z)};
  float inv = 1.0f - fac;
  return {src.x * inv + shifted.x * fac, src.y * inv + shifted.y * fac, src.z * inv + shifted.z * fac};
}
VN_INLINE float3 bright_contrast_rgb(float3 src, float bright, float contrast) {
  float a = 1.0f + contrast, b = bright - 0.5f * contrast;
  return {fmaxf(0.0f, a * src.x + b), fmaxf(0.0f, a * src.y + b), fmaxf(0.0f, a * src.z + b)};
}
VN_INLINE float3 curve_rgb(float3 rgb, const float *lut, int size, float low, float high, float fac, bool extrapolate) {
  auto channel = [&](int c, float value) {
    float x = (value - low) * (float(size - 1) / (high - low));
    float clipped = fmaxf(0.0f, fminf(float(size - 1), x));
    int lo = (int)floorf(clipped), hi = lo < size - 1 ? lo + 1 : size - 1;
    float t = clipped - (float)lo;
    const float *table = lut + c * size;
    float result = table[lo] * (1.0f - t) + table[hi] * t;
    if (extrapolate && x < 0.0f) result = table[0] + x * (table[1] - table[0]);
    if (extrapolate && x > float(size - 1)) result = table[size-1] + (x - float(size-1)) * (table[size-1] - table[size-2]);
    return fac == 1.0f ? result : value * (1.0f - fac) + result * fac;
  };
  return {channel(0, rgb.x), channel(1, rgb.y), channel(2, rgb.z)};
}
VN_INLINE float3 ramp_rgb(float fac, const float *lut, int size) {
  float x = fmaxf(0.0f, fminf(1.0f, fac)) * float(size - 1);
  int lo = (int)floorf(x), hi = lo < size - 1 ? lo + 1 : size - 1;
  float t = x - float(lo);
  float3 a = {lut[lo*3], lut[lo*3+1], lut[lo*3+2]}, b = {lut[hi*3], lut[hi*3+1], lut[hi*3+2]};
  return a * (1.0f - t) + b * t;
}
VN_INLINE float3 gamma_rgb(float3 rgb, float gamma, bool preserve_negative) {
  auto channel = [&](float v) { return gamma == 0.0f ? 1.0f : (v > 0.0f ? powf(v, gamma) : (preserve_negative ? v : powf(0.0f, gamma))); };
  return {channel(rgb.x), channel(rgb.y), channel(rgb.z)};
}
VN_INLINE float3 clamp_rgb(float3 rgb, float low, float high) {
  return {fminf(high, fmaxf(low, rgb.x)), fminf(high, fmaxf(low, rgb.y)), fminf(high, fmaxf(low, rgb.z))};
}
VN_DEVICE float3 eval_color_graph(
    const ColorGraphNode *nodes, int n_nodes, int output, float2 base_uv, float3 vc,
    float object_random) {
  float3 slots[COLOR_GRAPH_MAX_NODES];
  int n = n_nodes < COLOR_GRAPH_MAX_NODES ? n_nodes : COLOR_GRAPH_MAX_NODES;
  for (int i = 0; i < n; i++) {
    ColorGraphNode nd = nodes[i];
    unsigned int tag = nd.tag;
    const unsigned int *pp = nd.payload;
    if (tag == COLOR_NODE_CONST) {
      slots[i] = make_float3(bits_float(pp[0]),
                             bits_float(pp[1]),
                             bits_float(pp[2]));
    } else if (tag == COLOR_NODE_IMAGE_TEX) {
      unsigned long long ptr =
          ((unsigned long long)pp[1] << 32) | (unsigned long long)pp[0];
      const float *tex = (const float *)ptr;
      int w = (int)pp[2];
      int h = (int)pp[3];
      int ch = (int)pp[4];
      float m0 = bits_float(pp[5]);
      float m1 = bits_float(pp[6]);
      float m2 = bits_float(pp[7]);
      float m3 = bits_float(pp[8]);
      float m4 = bits_float(pp[9]);
      float m5 = bits_float(pp[10]);
      float2 uv = make_float2(m0 * base_uv.x + m1 * base_uv.y + m2,
                              m3 * base_uv.x + m4 * base_uv.y + m5);
      slots[i] = sample_rgba(tex, w, h, ch, uv);
    } else if (tag == COLOR_NODE_MIX) {
      int ia = (int)pp[0];
      int ib = (int)pp[1];
      unsigned int blend = pp[2];
      unsigned int fac_src = pp[3];
      float fac;
      if (fac_src == 0u) {
        fac = bits_float(pp[6]);
      } else {
        int ifac = (int)pp[4];
        float3 fv = slots[ifac];
        fac = 0.2126f * fv.x + 0.7152f * fv.y + 0.0722f * fv.z;
      }
      bool clamp_out = pp[5] != 0u;
      slots[i] = mix_blend_rgb(slots[ia], slots[ib], fac, blend, clamp_out);
    } else if (tag == COLOR_NODE_INVERT) {
      int iin = (int)pp[0];
      float fac = bits_float(pp[1]);
      slots[i] = invert_rgb(slots[iin], fac);
    } else if (tag == COLOR_NODE_MATH) {
      int iin = (int)pp[0];
      unsigned int op = pp[1];
      bool clamp_out = pp[2] != 0u;
      float b = bits_float(pp[3]);
      float c = bits_float(pp[4]);
      bool swap = pp[5] != 0u;
      float3 o = math_apply_rgb(slots[iin], op, b, c, swap);
      if (clamp_out) {
        o.x = fmaxf(0.0f, fminf(1.0f, o.x));
        o.y = fmaxf(0.0f, fminf(1.0f, o.y));
        o.z = fmaxf(0.0f, fminf(1.0f, o.z));
      }
      slots[i] = o;
    } else if (tag == COLOR_NODE_RGB_CURVE) {
      int iin = (int)pp[0];
      unsigned long long ptr =
          ((unsigned long long)pp[2] << 32) | (unsigned long long)pp[1];
      const float *lut = (const float *)ptr;
      slots[i] = curve_rgb(slots[iin], lut, 256, 0.0f, 1.0f, 1.0f, false);
    } else if (tag == COLOR_NODE_BRIGHT_CONTRAST) {
      int iin = (int)pp[0];
      float bright = bits_float(pp[1]);
      float contrast = bits_float(pp[2]);
      slots[i] = bright_contrast_rgb(slots[iin], bright, contrast);
    } else if (tag == COLOR_NODE_HUE_SAT) {
      int iin = (int)pp[0];
      float hue = bits_float(pp[1]);
      float sat = bits_float(pp[2]);
      float val = bits_float(pp[3]);
      float fac = bits_float(pp[4]);
      slots[i] = hue_sat_rgb(slots[iin], hue, sat, val, fac, false);
    } else if (tag == COLOR_NODE_VERTEX_COLOR) {
      slots[i] = vc;
    } else if (tag == COLOR_NODE_OBJECT_RANDOM) {
      slots[i] = make_float3(object_random, object_random, object_random);
    } else if (tag == COLOR_NODE_COLOR_RAMP) {
      // Payload: [in_slot, lut_ptr_lo, lut_ptr_hi, lut_len]. The host bakes
      // the ramp to `lut_len` linear-RGB stops; we look up by Fac.x with
      // bilinear interpolation across the table.
      int iin = (int)pp[0];
      unsigned long long ptr =
          ((unsigned long long)pp[2] << 32) | (unsigned long long)pp[1];
      const float *lut = (const float *)ptr;
      int len = (int)pp[3];
      slots[i] = ramp_rgb(slots[iin].x, lut, len);
    } else {
      slots[i] = make_float3(1.0f, 1.0f, 1.0f);
    }
  }
  if (output < 0 || output >= n) {
    return make_float3(1.0f, 1.0f, 1.0f);
  }
  return slots[output];
}


} // namespace vibrt_nodes
