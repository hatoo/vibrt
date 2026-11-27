VN_INLINE float3 bump(ShaderData sd, differential3 dP, float hc, float hx,
                      float hy, float strength, float distance,
                      float filter_width, bool invert, bool object_space) {
  SceneData data{filter_width, {}, &sd};
  float stack[24] = {};
  stack_store_float3(stack, 0, sd.N);
  stack[3] = distance;
  stack[4] = hc;
  stack[5] = hx;
  stack[6] = hy;
  stack[7] = strength;
  stack_store_float3(stack, 12, dP.dx);
  stack_store_float3(stack, 15, dP.dy);
  uint4 node{0,
             0u | (3u << 8) | (uint(invert) << 16) | (uint(object_space) << 24),
             4u | (5u << 8) | (6u << 16) | (7u << 24), 20u | (8u << 8)};
  svm_node_set_bump<0>(&data, &sd, stack, node, 0);
  return stack_load_float3(stack, 20);
}
VN_INLINE float3 normal_map(ShaderData sd, float3 color, float strength,
                            uint space, bool directx, bool original) {
  SceneData data{0, {}, &sd};
  float stack[8] = {color.x, color.y, color.z, strength};
  uint flags = space | (directx ? NODE_NORMAL_MAP_FLAG_DIRECTX : 0) |
               (original ? NODE_NORMAL_MAP_FLAG_ORIGINAL : 0);
  uint4 node{0, 0u | (3u << 8) | (4u << 16) | (flags << 24), 0, 1};
  svm_node_normal_map(&data, &sd, stack, node);
  return stack_load_float3(stack, 4);
}
