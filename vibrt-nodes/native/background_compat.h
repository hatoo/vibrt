// View the renderer's normalized scalar CDFs as Cycles' (density, CDF) pairs.
using namespace vibrt_cycles;
using float2 = vibrt_cycles::float2;
using float3 = vibrt_cycles::float3;
using vibrt_cycles::make_float2;
using vibrt_cycles::make_float3;
struct Background {
  int map_res_x, map_res_y;
};
struct Distribution {
  Background background;
  const float *marginal, *conditional;
};
using KernelGlobals = const Distribution *;
VN_INLINE float2 marginal(KernelGlobals kg, int i) {
  int h = kg->background.map_res_y;
  return make_float2(i == h ? 1 : (kg->marginal[i + 1] - kg->marginal[i]) * h,
                     kg->marginal[i]);
}
VN_INLINE float2 conditional(KernelGlobals kg, int i) {
  int w = kg->background.map_res_x;
  return make_float2(
      i % (w + 1) == w ? 1 : (kg->conditional[i + 1] - kg->conditional[i]) * w,
      kg->conditional[i]);
}
// The exported environment uses Z-up latitude and atan2(Y, X) longitude.
VN_INLINE float3 equirectangular_to_direction(float u, float v) {
  float theta = M_PI_F * v, phi = M_2PI_F * u;
  return make_float3(sinf(theta) * cosf(phi), sinf(theta) * sinf(phi),
                     cosf(theta));
}
VN_INLINE float2 direction_to_equirectangular(float3 d) {
  float u = atan2f(d.y, d.x) * M_1_2PI_F;
  return make_float2(u < 0 ? u + 1 : u,
                     atan2f(sqrtf(d.x * d.x + d.y * d.y), d.z) * M_1_PI_F);
}
#undef kernel_data
#define kernel_data (*kg)
#undef kernel_data_fetch
#define light_background_marginal_cdf marginal
#define light_background_conditional_cdf conditional
#define kernel_data_fetch(name, index) name(kg, index)
