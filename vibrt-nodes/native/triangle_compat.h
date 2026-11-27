// World-space geometry adapter; sampling and PDF evaluation are upstream Cycles.
using namespace vibrt_cycles;
using float3 = vibrt_cycles::float3;
using float2 = vibrt_cycles::float2;
using vibrt_cycles::make_float2;
using vibrt_cycles::make_float3;
using vibrt_cycles::max;
using vibrt_cycles::min;
struct Triangle {
  float3 vertices[3];
  bool two_sided;
};
using KernelGlobals = const Triangle *;
struct ShaderData {
  int object = 0, prim = 0;
  float time = 0;
  float3 P, wi, Ng;
};
struct LightSample {
  int object, prim, shader, type, group;
  float3 P, D, Ng;
  float t, pdf, pdf_selection, eval_fac;
};
constexpr int SD_OBJECT_NEGATIVE_SCALE = 1, SD_MIS_BACK = 1, SD_MIS_FRONT = 2;
constexpr int SHADER_MASK = 0, SHADER_USE_MIS = 0, LIGHT_TRIANGLE = 0;
struct ShaderFlags {
  int flags;
};
VN_INLINE bool triangle_world_space_vertices(KernelGlobals kg, int, int, float, float3 V[3]) {
  for (int i = 0; i < 3; ++i)
    V[i] = kg->vertices[i];
  return false;
}
VN_INLINE int triangle_fetch_object_flag(KernelGlobals, int) { return 0; }
VN_INLINE int triangle_fetch_tri_shader(KernelGlobals, int) { return 0; }
VN_INLINE ShaderFlags triangle_fetch_shaders(KernelGlobals kg, int) {
  return {SD_MIS_FRONT | (kg->two_sided ? SD_MIS_BACK : 0)};
}
VN_INLINE int object_lightgroup(KernelGlobals, int) { return 0; }
struct KernelData {
  struct {
    bool use_light_tree = true;
    float distribution_pdf_triangles = 0;
  } integrator;
};
#undef kernel_data
#undef kernel_data_fetch
#define kernel_data                                                                                \
  KernelData {}
#define kernel_data_fetch(name, index) triangle_fetch_##name(kg, index)
