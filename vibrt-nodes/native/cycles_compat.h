// Minimal scalar/vector platform adapter for the unmodified Cycles node math.
// No Blender scene or Python objects are accessed by these functions.
#pragma once
#ifndef __CUDACC__
#include <cfloat>
#include <cmath>
#include <cstring>
#define VN_INLINE inline
#define VN_DEVICE inline
#else
#define VN_INLINE __device__ __forceinline__
#define VN_DEVICE static __device__ __noinline__
#endif
#define ccl_device VN_DEVICE
#define ccl_device_inline VN_INLINE
#define ccl_device_forceinline VN_INLINE
#define ccl_device_noinline VN_DEVICE
#define ccl_device_noinline_cpu VN_DEVICE
#define ccl_private
#define ccl_ignore_integer_overflow
#define ccl_constant const
#ifdef __CUDACC__
#define ccl_inline_constant static __device__ __constant__
#else
#define ccl_inline_constant static const
#endif
#define kernel_assert(x) ((void)0)
#define CCL_NAMESPACE_BEGIN
#define CCL_NAMESPACE_END
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#define M_2PI_F 6.28318530717958647692f
#define M_PI_2_F 1.57079632679489661923f
#define M_1_PI_F 0.31830988618379067154f
#endif
#define M_SQRT2_F 1.41421356237309504880f
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.192092896e-7f
#define FLT_MAX 3.402823466e+38f
#endif

namespace vibrt_cycles {
using uint = unsigned int;
template <int N> struct Vec {
  float x, y, z, w;
  VN_INLINE float &operator[](int i) { return (&x)[i]; }
  VN_INLINE const float &operator[](int i) const { return (&x)[i]; }
};
using float2 = Vec<2>;
using float3 = Vec<3>;
using float4 = Vec<4>;
template <int N> struct IVec {
  int x, y, z, w;
  VN_INLINE int &operator[](int i) { return (&x)[i]; }
  VN_INLINE const int &operator[](int i) const { return (&x)[i]; }
};
using int2 = IVec<2>;
using int3 = IVec<3>;
using int4 = IVec<4>;
VN_INLINE int2 make_int2(int x, int y) { return {x, y, 0, 0}; }
VN_INLINE int3 make_int3(int x, int y, int z) { return {x, y, z, 0}; }
VN_INLINE int4 make_int4(int x, int y, int z, int w) { return {x, y, z, w}; }
VN_INLINE int2 make_int2(int x) { return {x, x, 0, 0}; }
VN_INLINE int3 make_int3(int x) { return {x, x, x, 0}; }
VN_INLINE int4 make_int4(int x) { return {x, x, x, x}; }
VN_INLINE int2 make_int2(float2 x) { return {(int)x.x, (int)x.y, 0, 0}; }
VN_INLINE int3 make_int3(float3 x) { return {(int)x.x, (int)x.y, (int)x.z, 0}; }
VN_INLINE int4 make_int4(float4 x) { return {(int)x.x, (int)x.y, (int)x.z, (int)x.w}; }
VN_INLINE int4 zero_int4() { return {0, 0, 0, 0}; }
#define VN_INT_OP(OP)                                                                              \
  template <int N> VN_INLINE IVec<N> operator OP(IVec<N> a, IVec<N> b) {                           \
    for (int i = 0; i < N; i++)                                                                    \
      a[i] = (int)((uint)a[i] OP(uint) b[i]);                                                      \
    return a;                                                                                      \
  }
VN_INT_OP(+)
VN_INT_OP(*)
VN_INT_OP(^)
VN_INT_OP(&)
#undef VN_INT_OP
template <int N> VN_INLINE IVec<N> operator>>(IVec<N> a, int b) {
  for (int i = 0; i < N; i++)
    a[i] >>= b;
  return a;
}
VN_INLINE float2 make_float2(float x, float y) { return {x, y, 0, 0}; }
VN_INLINE float2 make_float2(float x) { return {x, x, 0, 0}; }
VN_INLINE float3 make_float3(float x, float y, float z) { return {x, y, z, 0}; }
VN_INLINE float3 make_float3(float x) { return {x, x, x, 0}; }
VN_INLINE float3 make_float3(float4 v) { return {v.x, v.y, v.z, 0}; }
VN_INLINE float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
VN_INLINE float4 make_float4(float x) { return {x, x, x, x}; }
VN_INLINE float4 make_float4(float3 v, float w) { return {v.x, v.y, v.z, w}; }
VN_INLINE float4 make_float4(float3 v) { return {v.x, v.y, v.z, 0}; }
VN_INLINE float2 make_float2(int2 v) { return {(float)v.x, (float)v.y, 0, 0}; }
VN_INLINE float3 make_float3(int3 v) { return {(float)v.x, (float)v.y, (float)v.z, 0}; }
VN_INLINE float4 make_float4(int4 v) { return {(float)v.x, (float)v.y, (float)v.z, (float)v.w}; }
VN_INLINE float2 one_float2() { return make_float2(1); }
VN_INLINE float3 one_float3() { return make_float3(1); }
VN_INLINE float4 one_float4() { return make_float4(1); }
VN_INLINE float3 zero_float3() { return make_float3(0); }
VN_INLINE float4 zero_float4() { return make_float4(0); }
#define VN_VECTOR_OP(OP)                                                                           \
  template <int N> VN_INLINE Vec<N> operator OP(Vec<N> a, Vec<N> b) {                              \
    for (int i = 0; i < N; i++)                                                                    \
      a[i] = a[i] OP b[i];                                                                         \
    return a;                                                                                      \
  }                                                                                                \
  template <int N> VN_INLINE Vec<N> operator OP(Vec<N> a, float b) {                               \
    for (int i = 0; i < N; i++)                                                                    \
      a[i] = a[i] OP b;                                                                            \
    return a;                                                                                      \
  }                                                                                                \
  template <int N> VN_INLINE Vec<N> operator OP(float a, Vec<N> b) {                               \
    for (int i = 0; i < N; i++)                                                                    \
      b[i] = a OP b[i];                                                                            \
    return b;                                                                                      \
  }
VN_VECTOR_OP(+)
VN_VECTOR_OP(-)
VN_VECTOR_OP(*)
VN_VECTOR_OP(/)
#undef VN_VECTOR_OP
#define VN_VECTOR_ASSIGN(OP)                                                                       \
  template <int N, typename T> VN_INLINE Vec<N> &operator OP## = (Vec<N> & a, T b) {               \
    a = a OP b;                                                                                    \
    return a;                                                                                      \
  }
VN_VECTOR_ASSIGN(+)
VN_VECTOR_ASSIGN(-)
VN_VECTOR_ASSIGN(*)
VN_VECTOR_ASSIGN(/)
#undef VN_VECTOR_ASSIGN
template <int N> VN_INLINE Vec<N> operator-(Vec<N> a) { return a * -1.0f; }
template <int N> VN_INLINE bool operator==(Vec<N> a, Vec<N> b) {
  for (int i = 0; i < N; i++)
    if (a[i] != b[i])
      return false;
  return true;
}
template <int N> VN_INLINE bool operator!=(Vec<N> a, Vec<N> b) { return !(a == b); }
template <int N> VN_INLINE Vec<N> operator>=(Vec<N> a, Vec<N> b) {
  for (int i = 0; i < N; i++)
    a[i] = a[i] >= b[i] ? 1.0f : 0.0f;
  return a;
}
template <int N> VN_INLINE Vec<N> mask(Vec<N> m, Vec<N> a) { return a * m; }
template <int N> VN_INLINE Vec<N> fmod(Vec<N> a, float b) {
  for (int i = 0; i < N; i++)
    a[i] = fmodf(a[i], b);
  return a;
}
VN_INLINE int float_to_int(float a) { return (int)a; }
VN_INLINE int count_leading_zeros(uint a) {
  int n = 0;
  while (n < 32 && (a & 0x80000000u) == 0) {
    a <<= 1;
    n++;
  }
  return n;
}
template <typename T> VN_INLINE T min(T a, T b) { return a < b ? a : b; }
template <typename T> VN_INLINE T max(T a, T b) { return a > b ? a : b; }
VN_INLINE float clamp(float a, float lo, float hi) { return fminf(fmaxf(a, lo), hi); }
VN_INLINE float saturatef(float x) { return clamp(x, 0, 1); }
VN_INLINE float safe_divide(float a, float b) { return b != 0 ? a / b : 0; }
VN_INLINE float safe_modulo(float a, float b) { return b != 0 ? fmodf(a, b) : 0; }
VN_INLINE float safe_floored_modulo(float a, float b) { return b != 0 ? a - floorf(a / b) * b : 0; }
VN_INLINE float safe_fmod(float a, float b) { return safe_modulo(a, b); }
VN_INLINE float safe_powf(float a, float b) { return a < 0 && b != floorf(b) ? 0 : powf(a, b); }
VN_INLINE float safe_logf(float a, float b) {
  return a > 0 && b > 0 ? safe_divide(logf(a), logf(b)) : 0;
}
VN_INLINE float safe_sqrtf(float a) { return sqrtf(fmaxf(a, 0)); }
VN_INLINE float inversesqrtf(float a) { return a > 0 ? 1 / sqrtf(a) : 0; }
VN_INLINE float safe_asinf(float a) { return asinf(clamp(a, -1, 1)); }
VN_INLINE float safe_acosf(float a) { return acosf(clamp(a, -1, 1)); }
VN_INLINE float compatible_atan2(float a, float b) { return a == 0 && b == 0 ? 0 : atan2f(a, b); }
VN_INLINE float compatible_signf(float a) { return a > 0 ? 1 : (a < 0 ? -1 : 0); }
VN_INLINE float wrapf(float a, float hi, float lo) {
  float r = hi - lo;
  return r != 0 ? a - r * floorf((a - lo) / r) : lo;
}
VN_INLINE float pingpongf(float a, float b) {
  if (b == 0)
    return 0;
  return fabsf((a - b) - floorf((a - b) / (b * 2)) * (b * 2) - b);
}
VN_INLINE float smoothminf(float a, float b, float c) {
  if (c == 0)
    return fminf(a, b);
  float h = fmaxf(c - fabsf(a - b), 0) / c;
  return fminf(a, b) - h * h * h * c * (1.0f / 6.0f);
}
VN_INLINE float smoothstepf(float x) {
  x = saturatef(x);
  return x * x * (3 - 2 * x);
}
VN_INLINE float smoothstep(float lo, float hi, float x) {
  return smoothstepf(safe_divide(x - lo, hi - lo));
}
VN_INLINE float2 polar_to_cartesian(float r, float a) {
  return make_float2(r * cosf(a), r * sinf(a));
}
VN_INLINE float3 spherical_to_direction(float a, float b) {
  return make_float3(sinf(a) * cosf(b), sinf(a) * sinf(b), cosf(a));
}
VN_INLINE int floor_to_int(float x) { return (int)floorf(x); }
VN_INLINE float floorfrac(float x, int *i) {
  *i = floor_to_int(x);
  return x - *i;
}
VN_INLINE float fractf(float x) { return x - floorf(x); }
VN_INLINE float sqr(float x) { return x * x; }
VN_INLINE float __float_as_uint_dummy(float x) { return x; }
VN_INLINE uint __float_as_uint(float x) {
  uint u;
#ifdef __CUDACC__
  u = ::__float_as_uint(x);
#else
  std::memcpy(&u, &x, 4);
#endif
  return u;
}
VN_INLINE float __uint_as_float(uint u) {
#ifdef __CUDACC__
  return ::__uint_as_float(u);
#else
  float x;
  std::memcpy(&x, &u, 4);
  return x;
#endif
}
template <int N> VN_INLINE float dot(Vec<N> a, Vec<N> b) {
  float s = 0;
  for (int i = 0; i < N; i++)
    s += a[i] * b[i];
  return s;
}
template <int N> VN_INLINE float len_squared(Vec<N> a) { return dot(a, a); }
template <int N> VN_INLINE float len(Vec<N> a) { return sqrtf(dot(a, a)); }
VN_INLINE float len(float a) { return fabsf(a); }
template <int N> VN_INLINE float distance(Vec<N> a, Vec<N> b) { return len(a - b); }
template <int N> VN_INLINE Vec<N> safe_normalize(Vec<N> a) {
  float l = len(a);
  return l != 0 ? a / l : a * 0;
}
template <int N> VN_INLINE Vec<N> normalize(Vec<N> a) { return safe_normalize(a); }
template <typename T, typename U> VN_INLINE T mix(T a, T b, U f) { return a * (1.0f - f) + b * f; }
template <typename T> VN_INLINE T interp(T a, T b, float f) { return mix(a, b, f); }
template <int N> VN_INLINE float reduce_add(Vec<N> a) {
  float r = 0;
  for (int i = 0; i < N; i++)
    r += a[i];
  return r;
}
template <int N> VN_INLINE float reduce_max(Vec<N> a) {
  float r = a[0];
  for (int i = 1; i < N; i++)
    r = fmaxf(r, a[i]);
  return r;
}
#define VN_VECTOR_UNARY(NAME, FUNC)                                                                \
  template <int N> VN_INLINE Vec<N> NAME(Vec<N> a) {                                               \
    for (int i = 0; i < N; i++)                                                                    \
      a[i] = FUNC(a[i]);                                                                           \
    return a;                                                                                      \
  }
VN_VECTOR_UNARY(floor, floorf)
VN_VECTOR_UNARY(ceil, ceilf)
VN_VECTOR_UNARY(fabs, fabsf)
VN_VECTOR_UNARY(sin, sinf)
VN_VECTOR_UNARY(cos, cosf)
VN_VECTOR_UNARY(tan, tanf)
VN_VECTOR_UNARY(compatible_sign, compatible_signf)
#undef VN_VECTOR_UNARY
#define VN_VECTOR_BINARY(NAME, FUNC)                                                               \
  template <int N> VN_INLINE Vec<N> NAME(Vec<N> a, Vec<N> b) {                                     \
    for (int i = 0; i < N; i++)                                                                    \
      a[i] = FUNC(a[i], b[i]);                                                                     \
    return a;                                                                                      \
  }
VN_VECTOR_BINARY(min, fminf)
VN_VECTOR_BINARY(max, fmaxf)
VN_VECTOR_BINARY(safe_divide, safe_divide)
VN_VECTOR_BINARY(safe_fmod, safe_fmod)
VN_VECTOR_BINARY(safe_pow, safe_powf)
#undef VN_VECTOR_BINARY
template <int N> VN_INLINE Vec<N> safe_divide(Vec<N> a, float b) { return b != 0 ? a / b : a * 0; }
template <int N> VN_INLINE Vec<N> clamp(Vec<N> a, float lo, float hi) {
  for (int i = 0; i < N; i++)
    a[i] = clamp(a[i], lo, hi);
  return a;
}
template <int N> VN_INLINE Vec<N> power(Vec<N> a, float b) {
  for (int i = 0; i < N; i++)
    a[i] = powf(a[i], b);
  return a;
}
template <int N> VN_INLINE Vec<N> wrap(Vec<N> a, Vec<N> hi, Vec<N> lo) {
  for (int i = 0; i < N; i++)
    a[i] = wrapf(a[i], hi[i], lo[i]);
  return a;
}
VN_INLINE float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
VN_INLINE float3 project(float3 a, float3 b) { return b * safe_divide(dot(a, b), dot(b, b)); }
VN_INLINE float3 reflect(float3 a, float3 b) { return a - 2 * dot(a, b) * b; }
VN_INLINE float3 refract(float3 a, float3 b, float eta) {
  float d = dot(a, b), k = 1 - eta * eta * (1 - d * d);
  return k < 0 ? zero_float3() : eta * a - (eta * d + sqrtf(k)) * b;
}
VN_INLINE float3 faceforward(float3 a, float3 b, float3 c) { return dot(b, c) < 0 ? a : -a; }
VN_INLINE float3 rgb_to_hsv(float3 v) {
  auto r = vibrt_nodes::rgb_to_hsv_bl({v.x, v.y, v.z});
  return make_float3(r.x, r.y, r.z);
}
VN_INLINE float3 hsv_to_rgb(float3 v) {
  auto r = vibrt_nodes::hsv_to_rgb_bl({v.x, v.y, v.z});
  return make_float3(r.x, r.y, r.z);
}
VN_INLINE float3 saturate(float3 v) { return clamp(v, 0, 1); }
VN_INLINE float3 rgb_to_hsl(float3 c) {
  float3 h = rgb_to_hsv(c);
  float hi = fmaxf(c.x, fmaxf(c.y, c.z)), lo = fminf(c.x, fminf(c.y, c.z));
  float l = (hi + lo) * 0.5f;
  return make_float3(h.x, hi == lo ? 0 : (hi - lo) / (l > 0.5f ? 2 - hi - lo : hi + lo), l);
}
VN_INLINE float3 hsl_to_rgb(float3 h) {
  float chroma = (1 - fabsf(2 * h.z - 1)) * h.y;
  float3 c = hsv_to_rgb(make_float3(h.x, 1, 1));
  return c * chroma + (h.z - chroma * 0.5f);
}
} // namespace vibrt_cycles
