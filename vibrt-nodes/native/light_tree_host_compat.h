// The construction kernels see only bounds and energy, with no Blender objects.
#ifndef _MSC_VER
#define __forceinline inline
#endif
using float3 = vibrt_cycles::float3;
using vibrt_cycles::make_float3;
struct BoundBox {
  enum empty_t { empty = 0 };
  float3 min, max;
  BoundBox(empty_t) : min(make_float3(FLT_MAX)), max(make_float3(-FLT_MAX)) {}
  BoundBox(float3 lower, float3 upper) : min(lower), max(upper) {}
  void grow(float3 p) {
    min = vibrt_cycles::min(min, p);
    max = vibrt_cycles::max(max, p);
  }
  void grow(const BoundBox &b) {
    grow(b.min);
    grow(b.max);
  }
  float3 size() const { return max - min; }
  float area() const {
    auto d = size();
    return 2 * (d.x * d.y + d.y * d.z + d.z * d.x);
  }
};
inline float max4(float a, float b, float c, float d) {
  return fmaxf(fmaxf(a, b), fmaxf(c, d));
}
