// Plain 64-byte node layout shared with Rust; no Blender scene ownership.
struct KernelLightTreeNode {
  float lower[3], energy;
  float upper[3], theta_o;
  float axis[3], theta_e;
  int left, right, parent, emitter;
};
static_assert(sizeof(KernelLightTreeNode) == 64, "light tree ABI");
using KernelGlobals = const KernelLightTreeNode *;
#undef kernel_data_fetch
#define kernel_data_fetch(name, index) kg[index]
template <bool in_volume_segment>
VN_INLINE void light_tree_child_importance(KernelGlobals, float3 P, float3 N, float, bool,
                                           const KernelLightTreeNode *node, float &hi, float &lo) {
  KernelBoundingBox bbox = {make_float3(node->lower[0], node->lower[1], node->lower[2]),
                            make_float3(node->upper[0], node->upper[1], node->upper[2])};
  KernelBoundingCone cone = {make_float3(node->axis[0], node->axis[1], node->axis[2]),
                             node->theta_o, node->theta_e};
  float3 center = (bbox.min + bbox.max) * 0.5f;
  float distance;
  float3 delta = safe_normalize_len(center - P, &distance);
  if (distance == 0)
    delta = make_float3(0, 0, 1);
  float radius = len(bbox.max - center);
  distance = fmaxf(fmaxf(distance, 0.5f * radius), 1e-10f);
  // A volume event is a point, not an equiangular ray segment. A zero normal
  // removes the surface-incidence term. Mixed/bumped closures retain support.
  if (dot(N, N) == 0)
    N = delta;
  float cos_u = light_tree_cos_bound_subtended_angle(bbox, center, P);
  light_tree_importance<false>(N, true, delta, cos_u, cone, distance, distance, node->energy, 1, hi,
                               lo);
  hi = fmaxf(hi, 1e-5f * node->energy / (distance * distance));
}
