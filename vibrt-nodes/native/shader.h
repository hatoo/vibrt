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
// Cycles' color/melanin parametrizations, followed by its unmodified Chiang
// closure.
VN_INLINE float3 hair_absorption(float3 reflectance, float roughness) {
  return bsdf_principled_hair_sigma_from_reflectance(max(reflectance, make_float3(1e-20f)),
                                                    roughness);
}
VN_INLINE float3 hair_sigma(float3 color_value, float3 absorption, float radial,
                            float random, float random_color, float melanin,
                            float redness, float3 tint, int parametrization) {
  if (parametrization == 0)
    return hair_absorption(color_value, radial);
  if (parametrization == 1) {
    melanin *= (1 + 2 * (random - 0.5f) * clamp(random_color, 0, 1));
    melanin = -logf(fmaxf(1 - melanin, 0.0001f));
    return bsdf_principled_hair_sigma_from_concentration(melanin * (1 - redness),
                                                        melanin * redness) +
           hair_absorption(tint, radial);
  }
  return absorption;
}
VN_INLINE ChiangHairBSDF
chiang_setup(ShaderData &sd, float3 color_value, float3 absorption,
             float roughness, float radial, float offset, float ior, float coat,
             float random, float random_roughness, float random_color,
             float melanin, float redness, float3 tint, int parametrization) {
  float factor = 1 + 2 * (random - 0.5f) * random_roughness;
  ChiangHairBSDF b = {};
  b.v = roughness * factor;
  b.s = radial * factor;
  b.alpha = offset;
  b.eta = fmaxf(ior, 1.0001f);
  b.m0_roughness = 1 - clamp(coat, 0, 1);
  b.sigma = hair_sigma(color_value, absorption, b.s, random, random_color,
                       melanin, redness, tint, parametrization);
  bsdf_hair_chiang_setup(&sd, &b);
  return b;
}
// Cycles svm/bevel.h cubic radial profile (Apache-2.0), v5.1.1.
VN_INLINE float bevel_cubic_radius(float u) {
  return svm_bevel_cubic_quintic_root_find(u);
}
VN_INLINE float bevel_cubic_pdf(float radius, float distance) {
  return radius > 0 ? svm_bevel_cubic_pdf(radius, distance) : 0;
}
VN_INLINE float volume_phase_pdf(unsigned int type, const float *p, float g,
                                 float cosine) {
  if (type == 1)
    return phase_fournier_forand(
        cosine, phase_fournier_forand_coeffs(clamp(p[1], 0.001f, 0.5f),
                                             fmaxf(p[0], 1.001f)));
  if (type == 2)
    return phase_draine(cosine, g, fmaxf(p[2], 0.0f));
  if (type == 3)
    return phase_rayleigh(cosine);
  if (type == 4) {
    float hg, d, alpha, w;
    phase_mie_fitted_parameters(p[3], &hg, &d, &alpha, &w);
    return (1 - w) * phase_henyey_greenstein(cosine, hg) +
           w * phase_draine(cosine, d, alpha);
  }
  return phase_henyey_greenstein(cosine, g);
}
VN_INLINE float3 volume_phase_sample(unsigned int type, const float *p, float g,
                                     float u, float v, float *pdf) {
  float3 Z = make_float3(0, 0, 1);
  float2 random = make_float2(u, v);
  if (type == 1)
    return phase_fournier_forand_sample(
        Z,
        phase_fournier_forand_coeffs(clamp(p[1], 0.001f, 0.5f),
                                     fmaxf(p[0], 1.001f)),
        random, pdf);
  if (type == 2)
    return phase_draine_sample(Z, g, fmaxf(p[2], 0.0f), random, pdf);
  if (type == 3)
    return phase_rayleigh_sample(Z, random, pdf);
  if (type == 4) {
    float hg, d, alpha, w;
    phase_mie_fitted_parameters(p[3], &hg, &d, &alpha, &w);
    float3 direction;
    if (u < w)
      direction = phase_draine_sample(
          Z, d, alpha, make_float2(u / fmaxf(w, 1e-20f), v), pdf);
    else
      direction = phase_henyey_greenstein_sample(
          Z, hg, make_float2((u - w) / fmaxf(1 - w, 1e-20f), v), pdf);
    *pdf = volume_phase_pdf(type, p, g, direction.z);
    return direction;
  }
  return phase_henyey_greenstein_sample(Z, g, random, pdf);
}
// Scene access is supplied by the renderer. The standalone CPU API rejects
// these operations rather than inventing geometry or returning fallback values.
struct SceneServices {
  const void *vertex;
  int derivative;
};
#ifdef __CUDACC__
VN_INLINE float4 scene_node(const ShaderInstruction &node, const float4 *slots,
                            const float4 *ctx, const SceneServices &services);
VN_INLINE vibrt_scene_nodes::ShaderData normal_map_data(const float4 *ctx, float4 tangent,
                                                        bool named, const SceneServices &services);
#endif
VN_INLINE float luminance(float3 v) { return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z; }
VN_INLINE float fresnel(float cosi, float eta) {
  return fresnel_dielectric_cos(cosi, fmaxf(eta, 1e-5f));
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
VN_INLINE float4 image_sample_raw(const ShaderInstruction &n, float3 v) {
  vibrt_image::ImageView data{(const float *)n.data, (int)n.table_len, n.params[2] != 0, false};
  vibrt_image::KernelImageInfo info{n.width, n.height, (int)n.modes[1], (int)n.modes[0], &data};
  return vibrt_image::Interpolator::interp(info, v.x, v.y);
}
VN_INLINE float4 image_sample(const ShaderInstruction &n, float3 v) {
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

// Context ABI: 32 float4 values. Outputs are up to 64 float4 socket values.
// Host validation guarantees topological order, arity, mode and table bounds.
#ifndef VIBRT_SHADER_STACK_SIZE
#define VIBRT_SHADER_STACK_SIZE 256
#endif
enum EvalGroup { EVAL_ALL = 1, EVAL_BUMP = 2, EVAL_ALPHA = 4, EVAL_EMISSION = 8, EVAL_MIX = 16 };
VN_INLINE float4 finite_value(float4 r) {
  for (int ch = 0; ch < 4; ch++)
    if (!(fabsf(r[ch]) <= FLT_MAX)) r[ch] = 0;
  return r;
}
// Whole-program forms packed by shader.rs; image filtering remains shared with
// the general evaluator. No operand stack or instruction loop is needed.
VN_INLINE void evaluate_single(const ShaderInstruction &n, float4 uv, float4 *outputs) {
  float4 r = finite_value(uv);
  if ((n.op & 255) == 252)
    r = make_float4(n.params[0], n.params[1], n.params[2], n.params[3]);
#if (VIBRT_SHADER_OPS & (1ull << 25))
  if ((n.op & 255) == 254) {
    r = image_sample(n, xyz(r));
    if (n.modes[3]) r = scalar(r.w);
    r = finite_value(r);
    switch (n.inputs[2]) {
    case 1: r = scalar(luminance(xyz(r))); break;
    case 2: r = scalar((r.x + r.y + r.z) / 3); break;
    case 3: r = scalar(r.w); break;
    case 4: r = scalar(r.x); break;
    }
  }
#endif
  outputs[n.inputs[1]] = finite_value(r);
}
VN_INLINE void evaluate_vm(const ShaderInstruction *nodes, int count,
                           const float4 *ctx, float4 *outputs,
                           unsigned int requested = EVAL_ALL,
                           const SceneServices *services = nullptr) {
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
      if (m[2]) {
        float stack[21];
        for (int i = 0; i < 6; ++i)
          stack_store_float3(stack, 3 * i, xyz(A(i)));
        svm_node_vector_map_range(stack, 0, 3 | (6 << 8) | (9 << 16) | (12 << 24),
                                  15 | (m[1] << 8) | (m[0] << 16) | (18 << 24), 0);
        r = color(stack_load_float3(stack, 18));
      } else {
        float stack[7];
        for (int i = 0; i < 6; ++i)
          stack[i] = S(i);
        svm_node_map_range(0, stack, 0, 1 | (2 << 8) | (3 << 16) | (4 << 24),
                           m[0] | (5 << 8) | (6 << 16), 0);
        float value = stack[6];
        if (m[1] && m[0] < 2)
          value = clamp(value, fminf(S(3), S(4)), fmaxf(S(3), S(4)));
        r = scalar(value);
      }
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 14))
    case 14: {
      r = color(svm_mapping((NodeMappingType)m[0], V(0), V(1), V(2), V(3)));
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 15))
    case 15: {
      float stack[16];
      stack_store_float3(stack, 0, V(0));
      stack_store_float3(stack, 3, V(1));
      stack_store_float3(stack, 6, V(2));
      stack[9] = S(3);
      stack_store_float3(stack, 10, V(4));
      svm_node_vector_rotate(stack, m[0] | (10 << 16) | (m[1] << 24), 3 | (6 << 8) | (9 << 16), 13);
      r = color(stack_load_float3(stack, 13));
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
      vibrt_scene_nodes::ShaderData sd = {};
      sd.N = sd.Ng = sd.normal = xyz(ctx[4]);
      sd.flag = ctx[16].x != 0 ? vibrt_scene_nodes::SD_BACKFACING : 0;
      float4 tangent = make_float4(xyz(ctx[7]), ctx[17].x);
      sd.tangent = xyz(tangent);
      sd.sign = tangent.w;
      sd.world_basis = m[0] == 0;
      sd.object_to_world = ctx + 22; sd.world_to_object = ctx + 25;
#ifdef __CUDACC__
      if (services)
        sd = normal_map_data(ctx, tangent, false, *services);
      if (m[0] != 0)
        sd.world_basis = false;
#endif
      r = color(vibrt_scene_nodes::normal_map(sd, V(0), S(1), m[0],
                                             m[1] != 0, m[2] != 0));
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 48))
    case 48: {
      vibrt_scene_nodes::ShaderData sd = {};
      sd.N = sd.Ng = sd.normal = xyz(ctx[4]);
      sd.flag = ctx[16].x != 0 ? vibrt_scene_nodes::SD_BACKFACING : 0;
      float4 tangent = A(2);
      sd.tangent = xyz(tangent);
      sd.sign = tangent.w;
      sd.world_basis = true;

#ifdef __CUDACC__
      if (services)
        sd = normal_map_data(ctx, tangent, true, *services);
#endif
      r = color(vibrt_scene_nodes::normal_map(sd, V(0), S(1), 0,
                                             m[0] != 0, m[1] != 0));
      break;
    }
#endif
#if (VIBRT_SHADER_OPS & (3ull << 49))
    case 49:
    case 50: {
      float3 d;
      if ((n.op & 255) == 49) {
        d = V(3);
        if (m[0] == 1)
          d = vector_transform(d, ctx, 2, 0, 1);
        d *= (S(0) - S(1)) * S(2);
      } else {
        d = (V(0) - S(1)) * S(2);
        if (m[0] == 0) {
          float3 N = vector_transform(xyz(ctx[4]), ctx, 2, 0, 1), T = V(3);
          float3 B = safe_normalize(cross(N, T)) * A(3).w;
          d = T * d.x + N * d.y + B * d.z;
        }
      }
      if (m[0] != 2)
        d = vector_transform(d, ctx, 0, 1, 0);
      r = color(d);
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
#ifdef __CUDACC__
#if (VIBRT_SHADER_OPS & ((255ull << 40) | (1ull << 51)))
    case 40:
    case 41:
    case 42:
    case 43:
    case 44:
    case 45:
    case 46:
    case 47:
    case 51:
      r = scene_node(n, slots, ctx, *services);
      break;
#endif
#endif
    case 255:
      outputs[m[0]] = A(0);
      continue;
    }
    // Match shader-domain behavior at singular inputs without propagating NaNs.
    slots[n.op >> 16] = finite_value(r);
#undef A
#undef S
#undef V
  }
}
VN_INLINE void evaluate(const ShaderInstruction *nodes, int count, const float4 *ctx,
                        float4 *outputs, unsigned int requested = EVAL_ALL) {
  if (count == 1) {
    const ShaderInstruction &n = nodes[0];
    if ((n.op >> 8) & requested)
      evaluate_single(n, (n.op & 255) == 252 ? make_float4(0) : ctx[0], outputs);
  } else {
    evaluate_vm(nodes, count, ctx, outputs, requested);
  }
}
} // namespace vibrt_shader
