// Scene-independent adapter for the unmodified Cycles hair closure functions.
// Included inside vibrt_cycles after the numerical node helpers.
#define __HAIR__
#define SHADER_CLOSURE_BASE                                                    \
  float3 N;                                                                    \
  int type;                                                                    \
  Spectrum weight
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
  float padding[64];
};
struct ShaderData {
  float3 wi, dPdu, Ng, N;
  int type;
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
VN_INLINE float fresnel_dielectric_cos(float cosi, float eta) {
  float c = fabsf(cosi), g = eta * eta - 1 + c * c;
  if (g <= 0)
    return 1;
  g = sqrtf(g);
  float a = (g - c) / (g + c), b = (c * (g + c) - 1) / (c * (g - c) + 1);
  return 0.5f * a * a * (1 + b * b);
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
VN_INLINE float fresnel_dielectric(float c, float eta, float *cos_t) {
  float g = eta * eta - 1 + c * c;
  if (g <= 0) {
    *cos_t = 0;
    return 1;
  }
  *cos_t = -sqrtf(g) / eta;
  return fresnel_dielectric_cos(c, eta);
}
VN_INLINE float3 refract_angle(float3 incident, float3 normal, float cos_t,
                               float inv_eta) {
  return (inv_eta * dot(normal, incident) + cos_t) * normal -
         inv_eta * incident;
}
VN_INLINE void bsdf_transparent_setup(ShaderData *sd, Spectrum, uint) {
  sd->transparent = true;
}
