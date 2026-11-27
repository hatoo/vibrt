// Scene-free light data views. Renderer adapters supply photometric units.
using namespace vibrt_cycles;
using float2 = vibrt_cycles::float2;
using float3 = vibrt_cycles::float3;
using float4 = vibrt_cycles::float4;
using vibrt_cycles::make_float2;
using vibrt_cycles::make_float3;
using vibrt_cycles::make_float4;
using vibrt_cycles::sqr;
using vibrt_cycles::sqrt;
using uint32_t = unsigned int;
constexpr uint32_t PATH_RAY_MIS_HAD_TRANSMISSION = 1;
struct Transform {
  float3 x, y, z;
};
struct KernelSpotLight {
  float radius, eval_fac;
  bool is_sphere;
  float cos_half_spot_angle, spot_smooth, cos_half_larger_spread;
  float3 dir;
};
struct KernelLight {
  float3 co;
  Transform inverse;
  KernelSpotLight spot;
  struct {
    float one_minus_cosangle, eval_fac, angle, pdf;
  } distant;
};
struct LightEval {
  float eval_fac = 0, pdf = 0;
};
struct LightSample {
  float3 P, Ng, D;
  float t, pdf, eval_fac;
};
using KernelGlobals = int;
VN_INLINE Transform lamp_get_inverse_transform(KernelGlobals,
                                               const KernelLight *light) {
  return light->inverse;
}
VN_INLINE float3 transform_direction(const Transform *m, float3 v) {
  return make_float3(dot(m->x, v), dot(m->y, v), dot(m->z, v));
}
VN_INLINE float4 sqrt(float4 v) {
  return make_float4(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z), sqrtf(v.w));
}
VN_INLINE float4 sqr(float4 v) { return v * v; }
VN_INLINE float3 make_float3(float2 xy, float z) {
  return vibrt_cycles::make_float3(xy.x, xy.y, z);
}
VN_INLINE float reduce_min(float4 v) {
  return fminf(fminf(v.x, v.y), fminf(v.z, v.w));
}
