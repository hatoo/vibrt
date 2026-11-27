using namespace vibrt_cycles;
using float4 = vibrt_cycles::float4;
using uint64_t = unsigned long long;
using vibrt_cycles::__float_as_uint;
using KernelGlobals = int;
constexpr int PASS_UNUSED = -1;
struct Film {
  int pass_combined = 0, pass_adaptive_aux_buffer = 4, pass_sample_count = 8,
      pass_stride = 9;
  float exposure = 1;
};
struct Data {
  Film film;
};
#undef kernel_data
#define kernel_data                                                            \
  Data {}
VN_INLINE float4 kernel_read_pass_float4(const float *p) {
  return vibrt_cycles::make_float4(p[0], p[1], p[2], p[3]);
}
VN_INLINE uint popcount(uint value) {
#ifdef __CUDACC__
  return __popc(value);
#else
  value -= (value >> 1) & 0x55555555u;
  value = (value & 0x33333333u) + ((value >> 2) & 0x33333333u);
  return (((value + (value >> 4)) & 0x0f0f0f0fu) * 0x01010101u) >> 24;
#endif
}
