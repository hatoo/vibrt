// Small ABI adapters for upstream SVM bytecode and numerical utilities.
struct Transform {
  float4 x, y, z;
};
struct uint4 {
  uint x, y, z, w;
};
#ifndef UNLIKELY
#define UNLIKELY(x) (x)
#endif
#ifndef ccl_global
#define ccl_global
#endif
VN_INLINE float3 normalize_len(float3 v, float *l) {
  *l = len(v);
  return v / *l;
}
VN_INLINE float3 safe_normalize_len(float3 v, float *l) {
  *l = len(v);
  return *l > 0 ? v / *l : make_float3(0);
}
VN_INLINE float triangle_area(float3 a, float3 b, float3 c) {
  return 0.5f * len(cross(b - a, c - a));
}
VN_INLINE bool isequal(float3 a, float3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
// Match the scalar (non SIMD) intersection helpers used by Cycles.
VN_INLINE float ray_triangle_dot(float3 a, float3 b) { return dot(a, b); }
VN_INLINE float3 ray_triangle_cross(float3 a, float3 b) { return cross(a, b); }
VN_INLINE float ray_triangle_reciprocal(float x) { return 1.0f / x; }
VN_INLINE int __float_as_int(float x) { return (int)__float_as_uint(x); }
VN_INLINE float __int_as_float(int x) { return __uint_as_float((uint)x); }
VN_INLINE float fast_acosf(float x) { return acosf(clamp(x, -1, 1)); }
VN_INLINE float fast_atan2f(float y, float x) { return atan2f(y, x); }
template <int N> VN_INLINE Vec<N> clamp(Vec<N> x, Vec<N> a, Vec<N> b) {
  for (int i = 0; i < N; ++i)
    x[i] = clamp(x[i], a[i], b[i]);
  return x;
}
VN_INLINE uint reverse_integer_bits(uint x) {
  x = (x >> 16) | (x << 16);
  x = ((x & 0xff00ff00u) >> 8) | ((x & 0x00ff00ffu) << 8);
  x = ((x & 0xf0f0f0f0u) >> 4) | ((x & 0x0f0f0f0fu) << 4);
  x = ((x & 0xccccccccu) >> 2) | ((x & 0x33333333u) << 2);
  return ((x & 0xaaaaaaaau) >> 1) | ((x & 0x55555555u) << 1);
}
VN_INLINE void svm_unpack_node_uchar4(uint v, uint *a, uint *b, uint *c, uint *d) {
  *a = v & 255;
  *b = (v >> 8) & 255;
  *c = (v >> 16) & 255;
  *d = v >> 24;
}
VN_INLINE void svm_unpack_node_uchar3(uint v, uint *a, uint *b, uint *c) {
  uint unused;
  svm_unpack_node_uchar4(v, a, b, c, &unused);
}
VN_INLINE bool stack_valid(uint slot) { return slot != 255; }
VN_INLINE float stack_load_float(const float *s, uint i) { return s[i]; }
VN_INLINE float3 stack_load_float3(const float *s, uint i) {
  return make_float3(s[i], s[i + 1], s[i + 2]);
}
VN_INLINE float stack_load_float_default(const float *s, uint i, uint value) {
  return stack_valid(i) ? s[i] : __uint_as_float(value);
}
VN_INLINE void stack_store_float(float *s, uint i, float x) { s[i] = x; }
VN_INLINE void stack_store_float3(float *s, uint i, float3 v) {
  s[i] = v.x;
  s[i + 1] = v.y;
  s[i + 2] = v.z;
}
// The adapter supplies every socket on the stack; no bytecode defaults are read.
VN_INLINE uint4 read_node(KernelGlobals, int *offset) {
  ++*offset;
  return {0, 0, 0, 0};
}
