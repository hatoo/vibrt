// Scene-independent adapter for unmodified Cycles surface/hair closure functions.
// Included inside vibrt_cycles after the numerical node helpers.
#define __HAIR__
#define ccl_restrict __restrict
#define SHADER_CLOSURE_BASE                                                    \
  float3 N;                                                                    \
  int type;                                                                    \
  Spectrum weight; \
  float sample_weight
#define M_LN_2PI_F 1.83787706640934548356f
#define M_SQRT_PI_8_F 0.62665706865775012560f
#define M_2_PI_F 0.63661977236758134308f
using uint64_t = unsigned long long;
using uint32_t = unsigned int;
using Spectrum = float3;
using KernelGlobals = int;
struct ShaderClosure {
  float3 N;
  int type;
  Spectrum weight;
  float sample_weight;
  float padding[64];
};
struct ShaderData {
  float3 wi, dPdu, Ng, N;
  int type, flag;
  float v;
  uint lcg_state;
  int num_closure, num_closure_left;
  bool transparent;
};
enum {
  PRIMITIVE_CURVE = 3,
  PRIMITIVE_CURVE_RIBBON = 2,
  SD_BSDF = 1,
  SD_BSDF_HAS_EVAL = 2,
  SD_BSDF_HAS_TRANSMISSION = 4,
  LABEL_GLOSSY = 1,
  LABEL_REFLECT = 2,
  LABEL_TRANSMIT = 4,
  LABEL_NONE = 0,
  LABEL_DIFFUSE = 8,
  LABEL_SINGULAR = 16,
  SD_USE_BUMP_MAP_CORRECTION = 16,
  SD_BSDF_NEEDS_LCG = 8
};
VN_INLINE Spectrum zero_spectrum() { return make_float3(0); }
VN_INLINE Spectrum one_spectrum() { return make_float3(1); }
VN_INLINE Spectrum make_spectrum(float x) { return make_float3(x); }
VN_INLINE float spectrum_to_gray(KernelGlobals, Spectrum x) {
  return (x.x + x.y + x.z) / 3;
}
VN_INLINE float3 exp(float3 x) {
  return make_float3(expf(x.x), expf(x.y), expf(x.z));
}
VN_INLINE float sin_from_cos(float x) { return safe_sqrtf(1 - x * x); }
VN_INLINE float cos_from_sin(float x) { return sin_from_cos(x); }
VN_INLINE float pow20(float x) {
  float x4 = sqr(sqr(x));
  return sqr(sqr(x4)) * x4;
}
VN_INLINE float pow22(float x) { return pow20(x) * sqr(x); }
VN_INLINE float3 to_local(float3 p, float3 x, float3 y, float3 z) {
  return make_float3(dot(p, x), dot(p, y), dot(p, z));
}

VN_INLINE float3 sqrt(float3 x) {
  return make_float3(sqrtf(x.x), sqrtf(x.y), sqrtf(x.z));
}
VN_INLINE void fast_sincosf(float x, float *s, float *c) {
  *s = sinf(x);
  *c = cosf(x);
}
VN_INLINE float signf(float x) { return x > 0 ? 1 : x < 0 ? -1 : 0; }
VN_INLINE bool isnan_safe(float x) { return x != x; }
VN_INLINE float3 normalize(float3 x) { return safe_normalize(x); }
VN_INLINE bool is_zero(float3 x) { return dot(x, x) == 0; }
VN_INLINE bool isfinite_safe(float3 x) {
  return isfinite(x.x) && isfinite(x.y) && isfinite(x.z);
}
VN_INLINE float average(float3 x) { return (x.x + x.y + x.z) / 3; }
VN_INLINE float average(float2 x) { return (x.x + x.y) * 0.5f; }
VN_INLINE float lcg_step_float(uint *state) {
  *state = 1103515245u * (*state) + 12345u;
  return (float)*state * (1.0f / (float)0xFFFFFFFFu);
}


VN_INLINE void bsdf_transparent_setup(ShaderData *sd, Spectrum, uint) {
  sd->transparent = true;
}

#define FOREACH_SPECTRUM_CHANNEL(i) for (int i = 0; i < 3; ++i)
#define GET_SPECTRUM_CHANNEL(v, i) ((v)[i])
#define THIN_FILM_TABLE_SIZE 512
#define THINFILM_THICKNESS_CUTOFF 0.1f
#ifndef FLT_MIN
#define FLT_MIN 1.1754943508222875e-38f
#endif
#define M_PI_4_F 0.7853981633974483f
VN_INLINE float2 zero_float2() { return make_float2(0); }
VN_INLINE float2 make_float2(float3 x) { return make_float2(x.x, x.y); }
VN_INLINE float3 disk_to_hemisphere(float2 x) { return make_float3(x.x, x.y, safe_sqrtf(1 - dot(x, x))); }
VN_INLINE float sqrt(float x) { return sqrtf(x); }
VN_INLINE float3 make_spectrum(float3 x) { return x; }
VN_INLINE float3 rgb_to_spectrum(float3 x) { return x; }
VN_INLINE float3 safe_divide_color(float3 a, float3 b) { return safe_divide(a, b); }
VN_INLINE float3 safe_sqrt(float3 x) { return make_float3(safe_sqrtf(x.x), safe_sqrtf(x.y), safe_sqrtf(x.z)); }
VN_INLINE float3 sqr(float3 x) { return x * x; }
VN_INLINE float3 safe_normalize_fallback(float3 x, float3 fallback) { return is_zero(x) ? fallback : safe_normalize(x); }
template<int N> VN_INLINE bool isequal(Vec<N> a, Vec<N> b) { for (int i = 0; i < N; ++i) if (a[i] != b[i]) return false; return true; }
VN_INLINE float madd(float a, float b, float c) { return a * b + c; }
VN_INLINE float3 log(float3 x) { return make_float3(logf(x.x), logf(x.y), logf(x.z)); }
VN_INLINE float inverse_lerp(float a, float b, float x) { return safe_divide(x - a, b - a); }
VN_INLINE float fast_erff(float x) { return erff(x); }
