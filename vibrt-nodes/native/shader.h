// Typed material graph interpreter. CPU and CUDA execute this same code.
#pragma once
#include "cycles_nodes.h"

struct ShaderInstruction {
  unsigned int op;
  unsigned int inputs[12];
  unsigned int modes[4];
  float params[4];
  int width;
  int height;
  unsigned int table_len;
  unsigned long long data;
};
static_assert(sizeof(ShaderInstruction) == 104, "ShaderInstruction ABI");

#ifndef VIBRT_SHADER_OPS
#define VIBRT_SHADER_OPS (~0ull)
#endif

namespace vibrt_shader {
using namespace vibrt_cycles;
using vibrt_cycles::float2;
using vibrt_cycles::float3;
using vibrt_cycles::float4;
using vibrt_cycles::make_float2;
using vibrt_cycles::make_float3;
using vibrt_cycles::make_float4;
VN_INLINE float4 scalar(float v) { return make_float4(v, v, v, 1); }
VN_INLINE float4 color(float3 v) { return make_float4(v, 1); }
VN_INLINE float3 xyz(float4 v) { return make_float3(v); }
VN_INLINE float luminance(float3 v) { return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z; }
VN_INLINE float3 rotate(float3 p, float3 axis, float angle) {
  axis = safe_normalize(axis);
  float c = cosf(angle), s = sinf(angle);
  return p * c + cross(axis, p) * s + axis * dot(axis, p) * (1 - c);
}
VN_INLINE float3 euler(float3 v, float3 r, bool inverse = false) {
  if (inverse) {
    v = rotate(v, make_float3(0, 0, 1), -r.z);
    v = rotate(v, make_float3(0, 1, 0), -r.y);
    return rotate(v, make_float3(1, 0, 0), -r.x);
  }
  v = rotate(v, make_float3(1, 0, 0), r.x);
  v = rotate(v, make_float3(0, 1, 0), r.y);
  return rotate(v, make_float3(0, 0, 1), r.z);
}
VN_INLINE float fresnel(float cosi, float eta) {
  cosi = fabsf(cosi);
  eta = fmaxf(eta, 1e-5f);
  float g = eta * eta - 1 + cosi * cosi;
  if (g <= 0)
    return 1;
  g = sqrtf(g);
  float a = (g - cosi) / (g + cosi);
  float b = (cosi * (g + cosi) - 1) / (cosi * (g - cosi) + 1);
  return 0.5f * a * a * (1 + b * b);
}
VN_INLINE float4 lut_sample(const float *p, int count, float x, bool linear = true,
                            bool extend = false) {
  if (!linear) {
    int i = (int)floorf(clamp(x * (count - 1), 0.0f, (float)(count - 1)));
    return make_float4(p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]);
  }
  float q = x * (count - 1);
  int i = (int)floorf(clamp(q, 0.0f, (float)(count - 2)));
  float t = extend ? q - i : clamp(q - i, 0, 1);
  if (!linear)
    t = 0;
  return mix(make_float4(p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]),
             make_float4(p[4 * i + 4], p[4 * i + 5], p[4 * i + 6], p[4 * i + 7]), t);
}
VN_INLINE float3 transform(float3 v, const float4 *rows, bool point, bool transpose = false) {
  if (transpose)
    return xyz(rows[0]) * v.x + xyz(rows[1]) * v.y + xyz(rows[2]) * v.z;
  return make_float3(dot(xyz(rows[0]), v) + (point ? rows[0].w : 0),
                     dot(xyz(rows[1]), v) + (point ? rows[1].w : 0),
                     dot(xyz(rows[2]), v) + (point ? rows[2].w : 0));
}
VN_INLINE float3 vector_transform(float3 v, const float4 *ctx, int type, int from, int to) {
  bool point = type == 1, normal = type == 2;
  if (from == to)
    return v;
  if (from == 1)
    v = normal ? transform(v, ctx + 25, false, true) : transform(v, ctx + 22, point);
  if (from == 2) {
    if (point)
      v -= make_float3(ctx[28].w, ctx[29].w, ctx[30].w);
    v = transform(v, ctx + 28, false, true);
  }
  if (to == 1)
    v = normal ? transform(v, ctx + 22, false, true) : transform(v, ctx + 25, point);
  if (to == 2)
    v = transform(v, ctx + 28, point);
  return normal ? safe_normalize(v) : v;
}
VN_INLINE int tex_index(int i, int n, int extension) {
  if (extension == 0) {
    i %= n;
    return i < 0 ? i + n : i;
  }
  if (extension == 3) {
    i %= n * 2;
    if (i < 0)
      i += n * 2;
    return i < n ? i : 2 * n - 1 - i;
  }
  return min(max(i, 0), n - 1);
}
VN_INLINE float4 texel(const ShaderInstruction &n, int x, int y) {
  if (n.modes[1] == 2 && (x < 0 || y < 0 || x >= n.width || y >= n.height))
    return make_float4(0);
  x = tex_index(x, n.width, n.modes[1]);
  y = tex_index(y, n.height, n.modes[1]);
  const float *p = (const float *)n.data + 4 * (y * n.width + x);
  float a = n.params[2] != 0 ? p[3] : 1;
  return make_float4(p[0] * a, p[1] * a, p[2] * a, p[3]);
}
VN_DEVICE float4 image_sample_raw(const ShaderInstruction &n, float3 v) {
  if (n.modes[0] == 1)
    return texel(n, (int)floorf(v.x * n.width), (int)floorf(v.y * n.height));
  float x = v.x * n.width - 0.5f, y = v.y * n.height - 0.5f;
  int ix = (int)floorf(x), iy = (int)floorf(y);
  float tx = x - ix, ty = y - iy;
  if (n.modes[0] >= 2) {
    float wx[4] = {powf(1 - tx, 3) / 6, (3 * tx * tx * tx - 6 * tx * tx + 4) / 6,
                   (-3 * tx * tx * tx + 3 * tx * tx + 3 * tx + 1) / 6, tx * tx * tx / 6};
    float wy[4] = {powf(1 - ty, 3) / 6, (3 * ty * ty * ty - 6 * ty * ty + 4) / 6,
                   (-3 * ty * ty * ty + 3 * ty * ty + 3 * ty + 1) / 6, ty * ty * ty / 6};
    float4 r = make_float4(0);
    for (int j = 0; j < 4; j++)
      for (int i = 0; i < 4; i++)
        r += texel(n, ix + i - 1, iy + j - 1) * (wx[i] * wy[j]);
    return r;
  }
  return mix(mix(texel(n, ix, iy), texel(n, ix + 1, iy), tx),
             mix(texel(n, ix, iy + 1), texel(n, ix + 1, iy + 1), tx), ty);
}
VN_DEVICE float4 image_sample(const ShaderInstruction &n, float3 v) {
  float4 r = image_sample_raw(n, v);
  if (n.params[3] != 0 && r.w != 0 && r.w != 1) {
    float a = r.w;
    r = r / a;
    r.w = a;
  }
  return r;
}
VN_DEVICE float4 image_box(const ShaderInstruction &n, float3 co, float3 normal) {
  float3 N = fabs(normal);
  N = N / fmaxf(reduce_add(N), 1e-20f);
  float3 w = zero_float3();
  float blend = n.params[0], limit = 0.5f * (1 + blend);
  if (N.x > limit * (N.x + N.y) && N.x > limit * (N.x + N.z))
    w.x = 1;
  else if (N.y > limit * (N.x + N.y) && N.y > limit * (N.y + N.z))
    w.y = 1;
  else if (N.z > limit * (N.x + N.z) && N.z > limit * (N.y + N.z))
    w.z = 1;
  else if (blend > 0) {
    if (N.z < (1 - limit) * (N.y + N.x)) {
      w.x = saturatef((N.x / (N.x + N.y) - 0.5f * (1 - blend)) / blend);
      w.y = 1 - w.x;
    } else if (N.x < (1 - limit) * (N.y + N.z)) {
      w.y = saturatef((N.y / (N.y + N.z) - 0.5f * (1 - blend)) / blend);
      w.z = 1 - w.y;
    } else if (N.y < (1 - limit) * (N.x + N.z)) {
      w.x = saturatef((N.x / (N.x + N.z) - 0.5f * (1 - blend)) / blend);
      w.z = 1 - w.x;
    } else
      w = ((2 - limit) * N + (limit - 1)) / (2 * limit - 1);
  } else
    w.x = 1;
  float4 r = make_float4(0);
  if (w.x > 0)
    r += image_sample(n, make_float3(normal.x < 0 ? 1 - co.y : co.y, co.z, 0)) * w.x;
  if (w.y > 0)
    r += image_sample(n, make_float3(normal.y > 0 ? 1 - co.x : co.x, co.z, 0)) * w.y;
  if (w.z > 0)
    r += image_sample(n, make_float3(normal.z > 0 ? 1 - co.y : co.y, co.x, 0)) * w.z;
  return r;
}
template <typename T> VN_DEVICE float4 voronoi(const VoronoiParams &params, T p, int output) {
  if (params.feature == NODE_VORONOI_DISTANCE_TO_EDGE)
    return scalar(fractal_voronoi_distance_to_edge(params, p));
  if (params.feature == NODE_VORONOI_N_SPHERE_RADIUS)
    return scalar(voronoi_n_sphere_radius(params, p));
  VoronoiOutput o = fractal_voronoi_x_fx(params, p);
  if (output == 1)
    return color(o.color);
  if (output == 2)
    return color(xyz(o.position));
  if (output == 3)
    return scalar(o.position.w);
  return scalar(o.distance);
}

// Context ABI: 32 float4 values. Outputs are 32 float4 socket values.
// Host validation guarantees topological order, arity, mode and table bounds.
#ifndef VIBRT_SHADER_STACK_SIZE
#define VIBRT_SHADER_STACK_SIZE 256
#endif
enum EvalGroup { EVAL_ALL = 1, EVAL_BUMP = 2, EVAL_ALPHA = 4, EVAL_EMISSION = 8, EVAL_MIX = 16 };
VN_DEVICE void evaluate(const ShaderInstruction *nodes, int count, const float4 *ctx,
                        float4 *outputs, unsigned int requested = EVAL_ALL) {
  float4 slots[VIBRT_SHADER_STACK_SIZE];
  for (int i = 0; i < count; i++) {
    const ShaderInstruction &n = nodes[i];
    if (!((n.op >> 8) & requested))
      continue;
    const unsigned int *m = n.modes;
    const float *p = n.params;
#define A(k) slots[n.inputs[k]]
#define S(k) (A(k).x)
#define V(k) xyz(A(k))
    float4 r = make_float4(0, 0, 0, 1);
    switch (n.op & 255) {
#if (VIBRT_SHADER_OPS & (1ull << 0))
    case 0:
      r = make_float4(p[0], p[1], p[2], p[3]);
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 1))
    case 1:
      r = ctx[m[0]];
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 2))
    case 2:
      r = m[0] == 0   ? scalar(luminance(V(0)))
          : m[0] == 1 ? scalar((S(0) + A(0).y + A(0).z) / 3)
          : m[0] == 2 ? scalar(A(0).w)
                      : scalar(S(0));
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 3))
    case 3: {
      float v = svm_math((NodeMathType)m[0], S(0), S(1), S(2));
      r = scalar(m[1] ? saturatef(v) : v);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 4))
    case 4: {
      float v = 0;
      float3 c = zero_float3();
      svm_vector_math(&v, &c, (NodeVectorMathType)m[0], V(0), V(1), V(2), S(3));
      r = m[1] ? scalar(v) : color(c);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 5))
    case 5: {
      float4 f = m[1] == 2 ? A(0) : make_float4(S(0));
      if (m[2])
        f = clamp(f, 0, 1);
      r = m[1] ? mix(A(1), A(2), f) : color(svm_mix((NodeMix)m[0], f.x, V(1), V(2)));
      if (!m[1])
        r.w = A(1).w;
      if (m[3])
        r = clamp(r, 0, 1);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 6))
    case 6:
      r = color(mix(V(0), make_float3(1) - V(0), S(1)));
      r.w = A(0).w;
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 7))
    case 7: {
      float3 h = rgb_to_hsv(V(0));
      h.x += S(1) - 0.5f;
      h.y = clamp(h.y * S(2), 0, 1);
      h.z *= S(3);
      r = color(max(mix(V(0), hsv_to_rgb(h), S(4)), zero_float3()));
      r.w = A(0).w;
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 8))
    case 8:
      r = color(svm_math_gamma_color(V(0), S(1)));
      r.w = A(0).w;
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 9))
    case 9:
      r = color(svm_brightness_contrast(V(0), S(1), S(2)));
      r.w = A(0).w;
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 10))
    case 10:
      r = color(svm_combine_color((NodeCombSepColorType)m[0], make_float3(S(0), S(1), S(2))));
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 11))
    case 11:
      r = scalar(m[1] == 3 ? A(0).w : svm_separate_color((NodeCombSepColorType)m[0], V(0))[m[1]]);
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 12))
    case 12: {
      float lo = S(1), hi = S(2);
      if (m[0] && lo > hi) {
        float t = lo;
        lo = hi;
        hi = t;
      }
      r = scalar(clamp(S(0), lo, hi));
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 13))
    case 13: {
      float3 c = zero_float3();
      for (int ch = 0; ch < 3; ch++) {
        float lo = A(1)[ch], hi = A(2)[ch], to_lo = A(3)[ch], to_hi = A(4)[ch];
        float t = safe_divide(A(0)[ch] - lo, hi - lo);
        if (m[0] == 1)
          t = safe_divide(floorf(t * (A(5)[ch] + 1)), A(5)[ch]);
        if (m[0] >= 2) {
          t = saturatef(t);
          t = m[0] == 2 ? t * t * (3 - 2 * t) : t * t * t * (t * (t * 6 - 15) + 10);
        }
        c[ch] = to_lo + t * (to_hi - to_lo);
        if (m[1])
          c[ch] = clamp(c[ch], fminf(to_lo, to_hi), fmaxf(to_lo, to_hi));
      }
      r = color(c);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 14))
    case 14: {
      float3 v = V(0), loc = V(1), rot = V(2), scale = V(3);
      if (m[0] == 0)
        v = euler(v * scale, rot) + loc;
      else if (m[0] == 1)
        v = safe_divide(euler(v - loc, rot, true), scale);
      else if (m[0] == 2)
        v = euler(v * scale, rot);
      else
        v = safe_normalize(euler(safe_divide(v, scale), rot));
      r = color(v);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 15))
    case 15: {
      float3 v = V(0) - V(1), axis = V(2);
      float sign = m[1] ? -1.0f : 1.0f;
      if (m[0] == 4)
        v = euler(v, V(4), m[1] != 0);
      else {
        if (m[0] >= 1 && m[0] <= 3) {
          axis = zero_float3();
          axis[m[0] - 1] = 1;
        }
        v = rotate(v, axis, S(3) * sign);
      }
      r = color(v + V(1));
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 16))
    case 16: {
      float3 c = zero_float3(), co = V(0) * S(2);
      float w = S(1) * S(2), f = 0;
      float detail = clamp(S(3), 0, 15), rough = fmaxf(S(4), 0);
      bool want = m[3] != 0;
#define NOISE_ARGS detail, rough, S(5), S(6), S(7), S(8), m[1], m[2] != 0, want, &f, &c
      if (m[0] == 1)
        noise_texture_1d(w, NOISE_ARGS);
      if (m[0] == 2)
        noise_texture_2d(make_float2(co.x, co.y), NOISE_ARGS);
      if (m[0] == 3)
        noise_texture_3d(co, NOISE_ARGS);
      if (m[0] == 4)
        noise_texture_4d(make_float4(co, w), NOISE_ARGS);
#undef NOISE_ARGS
      r = want ? color(c) : scalar(f);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 17))
    case 17: {
      VoronoiParams q;
      q.scale = S(2);
      q.detail = clamp(S(3), 0, 15);
      q.roughness = clamp(S(4), 0, 1);
      q.lacunarity = S(5);
      q.smoothness = clamp(S(6) * 0.5f, 0, 0.5f);
      q.exponent = S(7);
      q.randomness = clamp(S(8), 0, 1);
      q.normalize = m[3] >= 5;
      q.feature = (NodeVoronoiFeature)m[1];
      q.metric = (NodeVoronoiDistanceMetric)m[2];
      q.max_distance =
          m[0] == 1 ? (0.5f + 0.5f * q.randomness)
                    : voronoi_distance(zero_float3(), make_float3(0.5f + 0.5f * q.randomness), q);
      if (m[0] == 2)
        q.max_distance =
            voronoi_distance(make_float2(0), make_float2(0.5f + 0.5f * q.randomness), q);
      if (m[0] == 4)
        q.max_distance =
            voronoi_distance(make_float4(0), make_float4(0.5f + 0.5f * q.randomness), q);
      if (m[1] == NODE_VORONOI_F2)
        q.max_distance *= 2;
      if (m[1] == NODE_VORONOI_DISTANCE_TO_EDGE)
        q.max_distance = 0.5f + 0.5f * q.randomness;
      float3 co = V(0) * q.scale;
      float w = S(1) * q.scale;
      int out = m[3] % 5;
      if (m[0] == 1)
        r = voronoi(q, w, out);
      if (m[0] == 2)
        r = voronoi(q, make_float2(co.x, co.y), out);
      if (m[0] == 3)
        r = voronoi(q, co, out);
      if (m[0] == 4)
        r = voronoi(q, make_float4(co, w), out);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 18))
    case 18:
      r = scalar(svm_wave((NodeWaveType)m[0], (NodeWaveBandsDirection)m[1],
                          (NodeWaveRingsDirection)m[2], (NodeWaveProfile)m[3], V(0) * S(1), S(2),
                          clamp(S(3), 0, 15), S(4), S(5), S(6)));
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 19))
    case 19: {
      float f = svm_checker(V(0) * S(3));
      r = m[0] ? scalar(f) : mix(A(2), A(1), f);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 20))
    case 20: {
      float2 v =
          svm_brick(V(0) * S(4), S(5), S(6), S(7), S(8), S(9), p[0], (int)p[1], p[2], (int)p[3]);
      r = m[0] ? scalar(v.y) : mix(mix(A(1), A(2), v.x), A(3), v.y);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 21))
    case 21:
      r = scalar(saturatef(svm_gradient(V(0), (NodeGradientType)m[0])));
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 22))
    case 22: {
      float3 c = svm_magic(V(0), S(1), m[0], S(2));
      r = m[1] ? scalar((c.x + c.y + c.z) / 3) : color(c);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 23))
    case 23: {
      float3 c = V(0);
      float4 v = make_float4(c, S(1));
      float f = 0;
      if (m[0] == 1) {
        f = hash_float_to_float(S(1));
        c = hash_float_to_float3(S(1));
      }
      if (m[0] == 2) {
        float2 u = make_float2(c.x, c.y);
        f = hash_float2_to_float(u);
        c = hash_float2_to_float3(u);
      }
      if (m[0] == 3) {
        f = hash_float3_to_float(c);
        c = hash_float3_to_float3(c);
      }
      if (m[0] == 4) {
        f = hash_float4_to_float(v);
        c = hash_float4_to_float3(v);
      }
      r = m[1] ? color(c) : scalar(f);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 24))
    case 24: {
      float3 co = V(0) * S(1);
      float f = fmaxf(S(2), 0.001f), iso = 1 - clamp(S(3), 0, 1);
      float2 ph;
      float dev;
      if (m[0] == 0) {
        ph = compute_2d_gabor_noise(make_float2(co.x, co.y), f, iso, S(4));
        dev = compute_2d_gabor_standard_deviation();
      } else {
        ph = compute_3d_gabor_noise(co, f, iso, safe_normalize(V(5)));
        dev = compute_3d_gabor_standard_deviation();
      }
      r = scalar(m[1] == 0   ? ph.y / (6 * dev) * 0.5f + 0.5f
                 : m[1] == 1 ? (atan2f(ph.y, ph.x) + M_PI_F) / M_2PI_F
                             : len(ph) / (6 * dev));
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 25))
    case 25: {
      float3 co = V(0);
      if (p[1] == 1) {
        co = safe_normalize(co);
        co = make_float3(-atan2f(co.y, co.x) / M_2PI_F + 0.5f,
                         atan2f(co.z, sqrtf(co.x * co.x + co.y * co.y)) / M_PI_F + 0.5f, 0);
      } else if (p[1] == 2) {
        co = safe_normalize(co);
        co = make_float3(atan2f(co.y, co.x) / M_2PI_F, acosf(clamp(co.z, -1, 1)) / M_PI_F, 0);
      } else if (p[1] == 3) {
        co = safe_normalize(co);
        float d = 2 * sqrtf(co.x * co.x + co.z * co.z + (co.y - 1) * (co.y - 1));
        co = make_float3(safe_divide(co.x, d) + 0.5f, safe_divide(co.z, d) + 0.5f, 0);
      } else if (m[2] >= 2) {
        co = co * 2 - 1;
        float l = len(co);
        co = make_float3((1 - atan2f(co.x, co.y) / M_PI_F) * 0.5f,
                         m[2] == 2 ? (l > 0 ? 1 - acosf(clamp(co.z / l, -1, 1)) / M_PI_F : 0)
                                   : co.z * 0.5f + 0.5f,
                         0);
      }
      if (m[2] == 1)
        r = image_box(n, co, xyz(ctx[13]));
      else
        r = image_sample(n, co);
      if (m[3])
        r = scalar(r.w);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 26))
    case 26: {
      const float *t = (const float *)n.data;
      int size = n.table_len / 4;
      float3 c = zero_float3();
      float scale = p[1] - p[0];
      for (int ch = 0; ch < 3; ch++) {
        float v = A(0)[ch];
        c[ch] = lut_sample(t, size, safe_divide(v - p[0], scale), true, m[1] != 0)[ch];
      }
      r = mix(A(0), color(c), S(1));
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 27))
    case 27:
      r = lut_sample((const float *)n.data, n.table_len / 4, S(0), m[0] == 0, false);
      if (m[1])
        r = scalar(r.w);
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 28))
    case 28:
      r = color(max(svm_math_blackbody_color_rec709(S(0)), zero_float3()));
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 29))
    case 29: {
      float3 v = svm_math_wavelength_color_xyz(S(0));
      r = color(max(make_float3(3.2404542f * v.x - 1.5371385f * v.y - 0.4985314f * v.z,
                                -0.969266f * v.x + 1.8760108f * v.y + 0.041556f * v.z,
                                0.0556434f * v.x - 0.2040259f * v.y + 1.0572252f * v.z) /
                        2.52f,
                    zero_float3()));
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 30))
    case 30:
      r = scalar(fresnel(dot(safe_normalize(V(1)), xyz(ctx[6])),
                         ctx[16].x != 0 ? 1 / fmaxf(S(0), 1e-5f) : S(0)));
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 31))
    case 31: {
      float facing = dot(V(1), xyz(ctx[6])), b = S(0);
      if (m[0] == 0) {
        float eta = fmaxf(1 - b, 1e-5f);
        r = scalar(fresnel(facing, ctx[16].x != 0 ? eta : 1 / eta));
      } else {
        facing = fabsf(facing);
        if (b != 0.5f) {
          b = clamp(b, 0, 1 - 1e-5f);
          b = b < 0.5f ? 2 * b : 0.5f / (1 - b);
          facing = powf(facing, b);
        }
        r = scalar(1 - facing);
      }
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 32))
    case 32: {
      float3 nrm = V(0) * 2 - 1;
      if (m[1])
        nrm.y = -nrm.y;
      if (m[0] >= 3) {
        nrm.y = -nrm.y;
        nrm.z = -nrm.z;
      }
      if (m[0] == 0) {
        float3 N = xyz(ctx[4]), T = xyz(ctx[7]), B = cross(N, T) * ctx[17].x;
        nrm.x *= S(1);
        nrm.y *= S(1);
        nrm.z = mix(1.0f, nrm.z, clamp(S(1), 0, 1));
        nrm = T * nrm.x + B * nrm.y + N * nrm.z;
      } else if (m[0] == 1 || m[0] == 3)
        nrm = vector_transform(nrm, ctx, 2, 1, 0);
      if (ctx[16].x != 0 && m[0] != 0)
        nrm = -nrm;
      nrm = safe_normalize(nrm);
      if (m[0] != 0)
        nrm = safe_normalize(mix(xyz(ctx[4]), nrm, fmaxf(S(1), 0)));
      if (dot(nrm, nrm) == 0)
        nrm = xyz(ctx[4]);
      r = color(nrm);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 33))
    case 33: {
      float3 nrm = safe_normalize(make_float3(p[0], p[1], p[2]));
      r = m[0] ? scalar(dot(safe_normalize(V(0)), nrm)) : color(nrm);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 34))
    case 34:
      r = color(vector_transform(V(0), ctx, m[0], m[1], m[2]));
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 35))
    case 35:
      r = scalar(1 / (1 + expf((S(0) - S(2)) * -S(1))));
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 36))
    case 36:
      r = m[0] == 0 ? color(safe_normalize(xyz(ctx[12])))
                    : scalar(m[0] == 1 ? ctx[12].z : len(xyz(ctx[12])));
      break;
#endif
#if (VIBRT_SHADER_OPS & (1ull << 37))
    case 37: {
      float d = ctx[18].x, v = S(0);
      if (m[0] == 1)
        v *= d;
      else if (m[0] == 2)
        v *= d * d;
      if (S(1) > 0)
        v *= d * d / (S(1) + d * d);
      r = scalar(v);
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 38))
    case 38: {
      float sides = fmaxf(S(1), 2), roundness = clamp(S(2), 0, 1);
      float2 co = make_float2(A(0).x, A(0).y);
      if (m[0] == 1)
        r = scalar(calculate_out_segment_id(sides, co));
      else {
        float4 v = calculate_out_variables(m[0] == 0, m[0] == 2, m[1] != 0, sides, roundness, co);
        r = m[0] == 0 ? make_float4(v.y, v.x, 0, 1) : scalar(m[0] == 2 ? v.z : v.w);
      }
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 39))
    case 39: {
      float3 t = xyz(ctx[7]), N = xyz(ctx[4]);
      if (m[0]) {
        float3 g = xyz(ctx[1]) - 0.5f;
        t = m[1] == 0   ? make_float3(0, -g.z, g.y)
            : m[1] == 1 ? make_float3(-g.z, 0, g.x)
                        : make_float3(-g.y, g.x, 0);
        t = vector_transform(t, ctx, 2, 1, 0);
      }
      r = color(cross(N, safe_normalize(cross(t, N))));
      break;
    }
#endif
    case 255:
      outputs[m[0]] = A(0);
      continue;
    }
    // Match shader-domain behavior at singular inputs without propagating NaNs.
    for (int ch = 0; ch < 4; ch++)
      if (!(fabsf(r[ch]) <= FLT_MAX))
        r[ch] = 0;
    slots[n.op >> 16] = r;
#undef A
#undef S
#undef V
  }
}
} // namespace vibrt_shader
