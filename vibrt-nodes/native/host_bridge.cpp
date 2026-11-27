#include <cstdint>
#include "cycles_host.h"

extern "C" void vibrt_filter_table(int type, float width, float *out) {
  auto table = vibrt_host::filter_table((vibrt_host::FilterType)type, width);
  std::copy(table.begin(), table.end(), out);
}

extern "C" int
vibrt_tree_partition(const vibrt_light_tree::KernelLightTreeNode *source,
                     unsigned int *indices, int count,
                     vibrt_light_tree::KernelLightTreeNode *node) {
  using namespace vibrt_host;
  std::vector<LightTreeEmitter> emitters(count);
  for (int i = 0; i < count; ++i) {
    const auto &s = source[indices[i]];
    auto &e = emitters[i];
    float3 lower = make_float3(s.lower[0], s.lower[1], s.lower[2]);
    float3 upper = make_float3(s.upper[0], s.upper[1], s.upper[2]);
    e.centroid = (lower + upper) * 0.5f;
    e.index = indices[i];
    e.measure = {
        BoundBox(lower, upper),
        {make_float3(s.axis[0], s.axis[1], s.axis[2]), s.theta_o, s.theta_e},
        s.energy};
  }
  LightTreeMeasure measure;
  LightTreeLightLink link;
  int middle = count / 2, dim = -1;
  LightTree().should_split(emitters.data(), 0, middle, count, measure, link,
                           dim);
  if (dim >= 0 && middle > 0 && middle < count) {
    std::nth_element(emitters.begin(), emitters.begin() + middle,
                     emitters.end(), [dim](const auto &a, const auto &b) {
                       return a.centroid[dim] == b.centroid[dim]
                                  ? a.index < b.index
                                  : a.centroid[dim] < b.centroid[dim];
                     });
  } else
    middle = count / 2;
  // Zero-energy emitters remain valid leaves even though Cycles excludes them
  // from the measure; traversal assigns them zero probability.
  if (measure.is_zero()) {
    for (const auto &e : emitters)
      measure.bbox.grow(e.measure.bbox);
    measure.bcone = {make_float3(0, 0, 1), M_PI_F, M_PI_2_F};
  }
  for (int c = 0; c < 3; ++c) {
    node->lower[c] = measure.bbox.min[c];
    node->upper[c] = measure.bbox.max[c];
    node->axis[c] = measure.bcone.axis[c];
  }
  node->theta_o = measure.bcone.theta_o;
  node->theta_e = measure.bcone.theta_e;
  node->energy = measure.energy;
  for (int i = 0; i < count; ++i)
    indices[i] = emitters[i].index;
  return middle;
}
