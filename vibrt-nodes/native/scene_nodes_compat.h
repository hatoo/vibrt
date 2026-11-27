using namespace vibrt_cycles;
using float3 = vibrt_cycles::float3;
using float4 = vibrt_cycles::float4;
using uint4 = vibrt_cycles::uint4;
using vibrt_cycles::__float_as_uint;
using vibrt_cycles::__uint_as_float;
using vibrt_cycles::make_float3;
using vibrt_differential::differential3;
using vibrt_differential::differential_from_compact;
constexpr int OBJECT_NONE = -1, PRIM_NONE = -1, PRIMITIVE_TRIANGLE = 1,
              PRIMITIVE_MOTION = 2;
constexpr int PRIMITIVE_CURVE = 4;
constexpr int SD_BACKFACING = 1, SHADER_SMOOTH_NORMAL = 1,
              SD_OBJECT_TRANSFORM_APPLIED = 1;
constexpr int ATTR_STD_NOT_FOUND = -1, ATTR_ELEMENT_NONE = 0,
              NODE_ATTR_FLOAT3 = 0;
constexpr int ATTR_STD_NORMAL_UNDISPLACED = 2, SVM_STACK_INVALID = 255;
struct ShaderData {
  float3 P, N, Ng, wi, tangent, normal, original_normal;
  float sign, dP, time;
  int prim = 0, type = PRIMITIVE_TRIANGLE, object = 0,
      shader = SHADER_SMOOTH_NORMAL;
  int flag = 0, object_flag = SD_OBJECT_TRANSFORM_APPLIED;
  bool world_basis = true, has_original = false;
  const float4 *object_to_world = nullptr, *world_to_object = nullptr;
};
struct SceneData {
  float filter_width;
  float3 vertices[3];
  const ShaderData *sd;
};
using KernelGlobals = const SceneData *;
VN_INLINE uint4 read_node(KernelGlobals kg, int *offset) {
  ++*offset;
  return {__float_as_uint(kg->filter_width), 0, 0, 0};
}
VN_INLINE void svm_unpack_node_uchar2(uint v, uint *a, uint *b) {
  *a = v & 255;
  *b = (v >> 8) & 255;
}
VN_INLINE float3 matrix_direction(const float4 *m, float3 v) {
  return m ? make_float3(dot(make_float3(m[0]), v), dot(make_float3(m[1]), v),
                         dot(make_float3(m[2]), v))
           : v;
}
VN_INLINE float3 matrix_transpose(const float4 *m, float3 v) {
  return m ? make_float3(m[0]) * v.x + make_float3(m[1]) * v.y +
                 make_float3(m[2]) * v.z
           : v;
}
VN_INLINE void object_normal_transform(KernelGlobals, ShaderData *sd,
                                       float3 *v) {
  *v = safe_normalize(
      sd->world_basis ? *v : matrix_transpose(sd->world_to_object, *v));
}
VN_INLINE void object_inverse_normal_transform(KernelGlobals, ShaderData *sd,
                                               float3 *v) {
  *v = sd->world_basis ? *v : matrix_transpose(sd->object_to_world, *v);
}
VN_INLINE void object_inverse_dir_transform(KernelGlobals, ShaderData *sd,
                                            float3 *v) {
  *v = sd->world_basis ? *v : matrix_direction(sd->world_to_object, *v);
}
VN_INLINE void object_position_transform(KernelGlobals, ShaderData *sd,
                                         float3 *v) {
  if (!sd->world_basis && sd->object_to_world)
    *v = matrix_direction(sd->object_to_world, *v) +
         make_float3(sd->object_to_world[0].w, sd->object_to_world[1].w,
                     sd->object_to_world[2].w);
}
struct AttributeDescriptor {
  int element, type, offset;
};
VN_INLINE AttributeDescriptor find_attribute(KernelGlobals, ShaderData *,
                                             uint index) {
  return {0, 0, (int)index};
}
VN_INLINE AttributeDescriptor find_attribute(KernelGlobals kg, int, int,
                                             int index) {
  return {0, 0, kg->sd->has_original ? index : ATTR_STD_NOT_FOUND};
}
template <class T> struct AttributeValue {
  T val;
};
template <class T>
VN_INLINE AttributeValue<T>
primitive_surface_attribute(KernelGlobals, ShaderData *, AttributeDescriptor,
                            bool = true, bool = true);
template <>
VN_INLINE AttributeValue<float>
primitive_surface_attribute(KernelGlobals, ShaderData *sd, AttributeDescriptor,
                            bool, bool) {
  return {sd->sign};
}
template <>
VN_INLINE AttributeValue<float3>
primitive_surface_attribute(KernelGlobals, ShaderData *sd,
                            AttributeDescriptor attr, bool, bool) {
  return {attr.offset == 2 ? sd->original_normal : sd->tangent};
}
VN_INLINE float3 triangle_smooth_normal_unnormalized_object_space(
    KernelGlobals, ShaderData *sd) {
  return sd->normal;
}
VN_INLINE void triangle_vertices(KernelGlobals kg, int, float3 *out) {
  for (int i = 0; i < 3; ++i)
    out[i] = kg->vertices[i];
}
VN_INLINE void motion_triangle_vertices(KernelGlobals kg, int, int, float,
                                        float3 *out) {
  triangle_vertices(kg, 0, out);
}
