VN_INLINE int sample(KernelGlobals nodes, float3 P, float3 N, float u, float *pdf) {
  *pdf = 1;
  if (!nodes || !(nodes[0].energy > 0)) {
    *pdf = 0;
    return -1;
  }
  int index = 0;
  while (nodes[index].emitter < 0) {
    const auto &node = nodes[index];
    float p;
    if (!get_left_probability<false>(nodes, P, N, 0, true, node.left, node.right, p)) {
      *pdf = 0;
      return -1;
    }
    if (u < p) {
      *pdf *= p;
      u /= p;
      index = node.left;
    } else {
      *pdf *= 1 - p;
      u = (u - p) / (1 - p);
      index = node.right;
    }
    u = fminf(u, 0.99999994f);
  }
  return nodes[index].emitter;
}
VN_INLINE float pdf(KernelGlobals nodes, int leaf, float3 P, float3 N) {
  if (!nodes || leaf < 0 || !(nodes[leaf].energy > 0))
    return 0;
  float result = 1;
  while (nodes[leaf].parent >= 0) {
    int parent = nodes[leaf].parent;
    const auto &node = nodes[parent];
    float p;
    if (!get_left_probability<false>(nodes, P, N, 0, true, node.left, node.right, p))
      return 0;
    result *= node.left == leaf ? p : 1 - p;
    leaf = parent;
  }
  return result;
}
