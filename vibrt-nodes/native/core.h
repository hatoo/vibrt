// Shared color-node implementation for CPU batches and CUDA shaders.
#pragma once
#ifndef __CUDACC__
#include <cmath>
#include <cstring>
#define VN_INLINE inline
#define VN_DEVICE inline
#else
#define VN_INLINE __device__ __forceinline__
#define VN_DEVICE static __device__
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
                                     int /*channels*/, float2 uv) {
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
    const float *p = &data[(y * w + x) * 4];
    return make_float3(p[0], p[1], p[2]);
  };
  float3 c00 = fetch(x0, y0);
  float3 c10 = fetch(x1, y0);
  float3 c01 = fetch(x0, y1);
  float3 c11 = fetch(x1, y1);
  float3 c0 = c00 * (1.0f - dx) + c10 * dx;
  float3 c1 = c01 * (1.0f - dx) + c11 * dx;
  return c0 * (1.0f - dy) + c1 * dy;
}

// RGB ↔ HSV helpers in the shape Blender uses (HSV channels in [0,1]).
VN_INLINE float3 rgb_to_hsv_bl(float3 c) {
  float cmax = fmaxf(c.x, fmaxf(c.y, c.z));
  float cmin = fminf(c.x, fminf(c.y, c.z));
  float v = cmax;
  float d = cmax - cmin;
  float s = cmax > 0.0f ? d / cmax : 0.0f;
  float h = 0.0f;
  if (d > 0.0f) {
    if (cmax == c.x) {
      h = (c.y - c.z) / d + (c.y < c.z ? 6.0f : 0.0f);
    } else if (cmax == c.y) {
      h = (c.z - c.x) / d + 2.0f;
    } else {
      h = (c.x - c.y) / d + 4.0f;
    }
    h *= (1.0f / 6.0f);
  }
  return make_float3(h, s, v);
}

VN_INLINE float3 hsv_to_rgb_bl(float3 c) {
  float h = c.x - floorf(c.x); // wrap to [0,1)
  float s = fmaxf(0.0f, fminf(1.0f, c.y));
  float v = c.z;
  float i = floorf(h * 6.0f);
  float f = h * 6.0f - i;
  float p = v * (1.0f - s);
  float q = v * (1.0f - f * s);
  float t = v * (1.0f - (1.0f - f) * s);
  int ii = ((int)i) % 6;
  if (ii < 0) ii += 6;
  switch (ii) {
    case 0: return make_float3(v, t, p);
    case 1: return make_float3(q, v, p);
    case 2: return make_float3(p, v, t);
    case 3: return make_float3(p, q, v);
    case 4: return make_float3(t, p, v);
    default: return make_float3(v, p, q);
  }
}

VN_INLINE float3 mix_blend_rgb(
    float3 a, float3 b, float fac, unsigned int blend, bool clamp_out) {
  float facm = 1.0f - fac;
  float3 out;
  switch (blend) {
    default:
    case 0: // MIX
      out = a * facm + b * fac;
      break;
    case 1: // MULTIPLY
      out = make_float3(a.x * (facm + b.x * fac),
                        a.y * (facm + b.y * fac),
                        a.z * (facm + b.z * fac));
      break;
    case 2: // ADD
      out = a + b * fac;
      break;
    case 3: // SUBTRACT
      out = a - b * fac;
      break;
    case 4: // SCREEN
      out = make_float3(1.0f, 1.0f, 1.0f) -
            (make_float3(facm, facm, facm) + (make_float3(1.0f, 1.0f, 1.0f) - b) * fac) *
                (make_float3(1.0f, 1.0f, 1.0f) - a);
      break;
    case 5: // DIVIDE — per-channel, guard against div-by-zero
      out.x = b.x == 0.0f ? a.x : a.x * facm + a.x / b.x * fac;
      out.y = b.y == 0.0f ? a.y : a.y * facm + a.y / b.y * fac;
      out.z = b.z == 0.0f ? a.z : a.z * facm + a.z / b.z * fac;
      break;
    case 6: { // DIFFERENCE
      float3 d = make_float3(fabsf(a.x - b.x), fabsf(a.y - b.y), fabsf(a.z - b.z));
      out = a * facm + d * fac;
      break;
    }
    case 7: // DARKEN
      out.x = a.x * facm + fminf(a.x, b.x) * fac;
      out.y = a.y * facm + fminf(a.y, b.y) * fac;
      out.z = a.z * facm + fminf(a.z, b.z) * fac;
      break;
    case 8: // LIGHTEN (Blender's asymmetric form)
      out.x = fmaxf(a.x, b.x * fac);
      out.y = fmaxf(a.y, b.y * fac);
      out.z = fmaxf(a.z, b.z * fac);
      break;
    case 9: { // OVERLAY per-channel
      auto ov = [&](float ax, float bx) {
        return ax < 0.5f ? ax * (facm + 2.0f * fac * bx)
                         : 1.0f - (facm + 2.0f * fac * (1.0f - bx)) * (1.0f - ax);
      };
      out.x = ov(a.x, b.x);
      out.y = ov(a.y, b.y);
      out.z = ov(a.z, b.z);
      break;
    }
    case 10: { // SOFT_LIGHT
      float3 scr = make_float3(1.0f - (1.0f - b.x) * (1.0f - a.x),
                               1.0f - (1.0f - b.y) * (1.0f - a.y),
                               1.0f - (1.0f - b.z) * (1.0f - a.z));
      float3 inner = make_float3(
          (1.0f - a.x) * b.x * a.x + a.x * scr.x,
          (1.0f - a.y) * b.y * a.y + a.y * scr.y,
          (1.0f - a.z) * b.z * a.z + a.z * scr.z);
      out = a * facm + inner * fac;
      break;
    }
    case 12: // HUE
    case 13: // SATURATION
    case 14: // COLOR
    case 15: { // VALUE
      float3 ha = rgb_to_hsv_bl(a), hb = rgb_to_hsv_bl(b), target = ha;
      bool active = true;
      if (blend == 12) { target.x = hb.x; active = hb.y != 0.0f; }
      if (blend == 13) { target.y = hb.y; active = ha.y != 0.0f; }
      if (blend == 14) { target.x = hb.x; target.y = hb.y; active = hb.y != 0.0f; }
      if (blend == 15) target.z = hb.z;
      out = active ? a * facm + hsv_to_rgb_bl(target) * fac : a;
      break;
    }
    case 11: // LINEAR_LIGHT
      out = a + (b * 2.0f - make_float3(1.0f, 1.0f, 1.0f)) * fac;
      break;
  }
  if (clamp_out) {
    out.x = fmaxf(0.0f, fminf(1.0f, out.x));
    out.y = fmaxf(0.0f, fminf(1.0f, out.y));
    out.z = fmaxf(0.0f, fminf(1.0f, out.z));
  }
  return out;
}

VN_INLINE float3 math_apply_rgb(
    float3 inp, unsigned int op, float b, float c, bool swap) {
  // `swap` flips operand order for non-commutative ops so the node can act as
  // `b OP input` instead of `input OP b` — mirrors Cycles ShaderNodeMath with
  // the texture chain on input[1]. Commutative ops ignore swap.
  float3 o;
  switch (op) {
    default:
    case 0: // ADD
      o = make_float3(inp.x + b, inp.y + b, inp.z + b);
      break;
    case 1: // SUBTRACT
      if (swap) {
        o = make_float3(b - inp.x, b - inp.y, b - inp.z);
      } else {
        o = make_float3(inp.x - b, inp.y - b, inp.z - b);
      }
      break;
    case 2: // MULTIPLY
      o = make_float3(inp.x * b, inp.y * b, inp.z * b);
      break;
    case 3: // DIVIDE
      if (swap) {
        // b / inp, per channel. Cycles returns 0 when the denominator is 0.
        o = make_float3(inp.x == 0.0f ? 0.0f : b / inp.x,
                        inp.y == 0.0f ? 0.0f : b / inp.y,
                        inp.z == 0.0f ? 0.0f : b / inp.z);
      } else {
        o = make_float3(b == 0.0f ? 0.0f : inp.x / b,
                        b == 0.0f ? 0.0f : inp.y / b,
                        b == 0.0f ? 0.0f : inp.z / b);
      }
      break;
    case 4: // POWER
      if (swap) {
        float base = fmaxf(b, 0.0f);
        o = make_float3(powf(base, inp.x), powf(base, inp.y), powf(base, inp.z));
      } else {
        o = make_float3(powf(fmaxf(inp.x, 0.0f), b),
                        powf(fmaxf(inp.y, 0.0f), b),
                        powf(fmaxf(inp.z, 0.0f), b));
      }
      break;
    case 5: // MULTIPLY_ADD
      o = make_float3(inp.x * b + c, inp.y * b + c, inp.z * b + c);
      break;
    case 6: // MINIMUM
      o = make_float3(fminf(inp.x, b), fminf(inp.y, b), fminf(inp.z, b));
      break;
    case 7: // MAXIMUM
      o = make_float3(fmaxf(inp.x, b), fmaxf(inp.y, b), fmaxf(inp.z, b));
      break;
  }
  return o;
}


// Scalar constant folding preserves the exporter's established edge cases;
// the per-channel image Math path above retains its original eight operations.
VN_INLINE float scalar_math(float a, unsigned int op, float b, float c) {
  switch (op) {
    case 4: return a >= 0.0f || floorf(b) == b ? powf(a, b) : 0.0f;
    case 8: return a > 0 && b > 0 && b != 1 ? logf(a) / logf(b) : nanf("");
    case 9: return a < b ? 1.0f : 0.0f;
    case 10: return a > b ? 1.0f : 0.0f;
    case 11: return fabsf(a - b) <= c ? 1.0f : 0.0f;
    case 12: return b != 0 ? fmodf(a, b) : 0.0f;
    case 13: return fminf(a, b);
    case 14: return fmaxf(a, b);
    case 15: return atan2f(a, b);
    case 16: return sqrtf(fmaxf(0.0f, a));
    case 17: return a > 0 ? 1.0f / sqrtf(a) : 0.0f;
    case 18: return fabsf(a);
    case 19: return expf(a);
    case 20: return float((a > 0) - (a < 0));
    case 21: return rintf(a);
    case 22: return floorf(a);
    case 23: return ceilf(a);
    case 24: return a - floorf(a);
    case 25: return truncf(a);
    case 26: return sinf(a);
    case 27: return cosf(a);
    case 28: return tanf(a);
    case 29: return asinf(fmaxf(-1.0f, fminf(1.0f, a)));
    case 30: return acosf(fmaxf(-1.0f, fminf(1.0f, a)));
    case 31: return atanf(a);
    case 32: return sinhf(a);
    case 33: return coshf(a);
    case 34: return tanhf(a);
    case 35: return a * 0.01745329251994329577f;
    case 36: return a * 57.295779513082320877f;
    default: return math_apply_rgb({a, a, a}, op, b, c, false).x;
  }
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
#undef VN_INLINE
#undef VN_DEVICE
