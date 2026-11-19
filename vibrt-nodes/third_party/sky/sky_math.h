// Minimal vector arithmetic and CPU scheduling supplied by vibrt.
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <thread>
#include <type_traits>
#include <vector>

constexpr float M_PI_F = 3.14159265358979323846f;
constexpr float M_PI_2_F = M_PI_F / 2.0f;
constexpr float M_2PI_F = 2.0f * M_PI_F;
constexpr float M_1_PI_F = 1.0f / M_PI_F;
constexpr float M_4PI_F = 4.0f * M_PI_F;
constexpr float M_1_4PI_F = 1.0f / M_4PI_F;

struct float2 {
  static constexpr int size = 2;
  float x, y;
  float2() = default;
  explicit float2(float a) : x(a), y(a) {}
  float2(float a, float b) : x(a), y(b) {}
  float &operator[](int i) { return i == 0 ? x : y; }
  float operator[](int i) const { return i == 0 ? x : y; }
};
struct float3 {
  static constexpr int size = 3;
  float x, y, z;
  float3() = default;
  explicit float3(float a) : x(a), y(a), z(a) {}
  float3(float a, float b, float c) : x(a), y(b), z(c) {}
  float &operator[](int i) { return i == 0 ? x : i == 1 ? y : z; }
  float operator[](int i) const { return i == 0 ? x : i == 1 ? y : z; }
  float length_squared() const { return x*x + y*y + z*z; }
  float length() const { return std::sqrt(length_squared()); }
};
struct float4 {
  static constexpr int size = 4;
  float x, y, z, w;
  float4() = default;
  explicit float4(float a) : x(a), y(a), z(a), w(a) {}
  float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
  float &operator[](int i) { return i == 0 ? x : i == 1 ? y : i == 2 ? z : w; }
  float operator[](int i) const { return i == 0 ? x : i == 1 ? y : i == 2 ? z : w; }
};
template<class V> using Vector = std::enable_if_t<
    std::is_same_v<V, float2> || std::is_same_v<V, float3> || std::is_same_v<V, float4>, V>;
#define VIBRT_VECTOR_OP(OP) \
  template<class V> Vector<V> operator OP(const V &a, const V &b) { \
    V out; for (int i = 0; i < V::size; ++i) out[i] = a[i] OP b[i]; return out; } \
  template<class V> Vector<V> operator OP(const V &a, float b) { return a OP V(b); } \
  template<class V> Vector<V> operator OP(float a, const V &b) { return V(a) OP b; }
VIBRT_VECTOR_OP(+)
VIBRT_VECTOR_OP(-)
VIBRT_VECTOR_OP(*)
VIBRT_VECTOR_OP(/)
#undef VIBRT_VECTOR_OP
template<class V> Vector<V> operator-(const V &a) { return V(0.0f) - a; }
template<class V> V &operator+=(V &a, const V &b) { a = a + b; return a; }
template<class V> V &operator*=(V &a, const V &b) { a = a * b; return a; }
template<class V> V &operator*=(V &a, float b) { a = a * b; return a; }
inline float2 make_float2(float x, float y) { return {x, y}; }
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
template<class V> float dot(const V &a, const V &b) {
  float value = a[0] * b[0];
  for (int i = 1; i < V::size; ++i) value += a[i] * b[i];
  return value;
}
inline float len_squared(float3 v) { return v.length_squared(); }
inline float len(float3 v) { return v.length(); }
inline float3 normalize(float3 v) { return v / len(v); }
inline float distance(float3 a, float3 b) { return len(a-b); }
inline float reduce_add(float3 v) { return v.x + v.y + v.z; }
inline float4 exp(float4 v) { return {expf(v.x), expf(v.y), expf(v.z), expf(v.w)}; }
inline float4 max(float4 v, float x) {
  return {std::max(v.x,x), std::max(v.y,x), std::max(v.z,x), std::max(v.w,x)};
}
inline float clamp(float x, float low, float high) { return std::clamp(x, low, high); }
inline float saturate(float x) { return clamp(x, 0.0f, 1.0f); }
inline float sqr(float x) { return x*x; }
inline float safe_sqrtf(float x) { return sqrtf(std::max(0.0f, x)); }
template<class T> T mix(T a, T b, float t) { return a + t * (b-a); }
inline float3 sun_direction(float cos_theta) { return {-safe_sqrtf(1-cos_theta*cos_theta), 0, cos_theta}; }
inline float ray_sphere_intersection(float3 origin, float3 direction, float radius) {
  const float projection = dot(origin, direction);
  const float offset = dot(origin, origin) - radius*radius;
  const float discriminant = projection*projection - offset;
  if ((offset > 0 && projection > 0) || discriminant < 0) return -1.0f;
  const float root = sqrtf(discriminant);
  return discriminant >= projection*projection ? -projection + root : -projection - root;
}
template<class F> void SKY_parallel_for(size_t begin, size_t end, size_t grain, const F &body) {
  if (begin >= end) return;
  const size_t count = std::min<size_t>({16, std::max(1u, std::thread::hardware_concurrency()),
                                       (end-begin+grain-1)/grain});
  std::atomic<size_t> next{begin};
  auto work = [&]() {
    for (;;) {
      const size_t first = next.fetch_add(grain, std::memory_order_relaxed);
      if (first >= end) return;
      body(first, std::min(end, first+grain));
    }
  };
  std::vector<std::thread> workers;
  try {
    for (size_t i = 1; i < count; ++i) workers.emplace_back(work);
    work();
  } catch (...) {
    for (auto &worker : workers) worker.join();
    throw;
  }
  for (auto &worker : workers) worker.join();
}
