extern "C" __global__ void __closesthit__ch() {
  fill_path_vertex(get_path_vertex());
}

extern "C" __global__ void __closesthit__shadow() {
  // Not used when TERMINATE_ON_FIRST_HIT is set — kept to satisfy SBT.
}

// Legacy alpha-mask cutouts. Colored transparent shadows are evaluated in
// closest-hit order by the shadow traversal, using the material closures.
extern "C" __global__ void __anyhit__ah() {
  HitGroupData *hg = (HitGroupData *)optixGetSbtDataPointer();
  unsigned int prim = optixGetPrimitiveIndex();
  PrincipledGpu *m = hg->mat;
  if (hg->material_indices != nullptr && hg->num_materials > 0) {
    unsigned int mi = hg->material_indices[prim];
    if ((int)mi < hg->num_materials)
      m = hg->materials[mi];
  }
  if (m->alpha_threshold <= 0.0f || m->base_color_tex == nullptr)
    return;
  // Recompute UV from barycentrics.
  if (hg->uvs == nullptr)
    return;
  int i0 = hg->indices[prim * 3 + 0];
  int i1 = hg->indices[prim * 3 + 1];
  int i2 = hg->indices[prim * 3 + 2];
  float2 bary = hg->curve_radii ? make_float2(optixGetCurveParameter(), 0) : optixGetTriangleBarycentrics();
  float b0 = 1.0f - bary.x - bary.y;
  float2 uv0 = make_float2(hg->uvs[i0 * 2 + 0], hg->uvs[i0 * 2 + 1]);
  float2 uv1 = make_float2(hg->uvs[i1 * 2 + 0], hg->uvs[i1 * 2 + 1]);
  float2 uv2 = make_float2(hg->uvs[i2 * 2 + 0], hg->uvs[i2 * 2 + 1]);
  float2 uv =
      make_float2(b0 * uv0.x + bary.x * uv1.x + bary.y * uv2.x,
                  b0 * uv0.y + bary.x * uv1.y + bary.y * uv2.y);
  const float *M = m->uv_transform;
  uv = make_float2(M[0] * uv.x + M[1] * uv.y + M[2],
                   M[3] * uv.x + M[4] * uv.y + M[5]);
  int w = m->base_color_tex_w, h = m->base_color_tex_h;
  if (w <= 0 || h <= 0)
    return;
  float alpha = sample_image(m->base_color_tex, w, h, m->base_color_tex_channels, uv.x, uv.y).w;
  if (alpha < m->alpha_threshold)
    optixIgnoreIntersection();
}

extern "C" __global__ void __miss__ms() {
  PathVertex *v = get_path_vertex();
  v->hit = 0;
}

extern "C" __global__ void __miss__shadow() { optixSetPayload_0(1u); }

// Query intersections ignore material alpha, preserving raw geometry semantics.
extern "C" __global__ void __anyhit__query() {
  QueryPayload *p = (QueryPayload *)unpack_ptr(optixGetPayload_0(), optixGetPayload_1());
  unsigned int instance = optixGetInstanceId();
  if ((p->only_local && instance != p->instance) ||
      (p->skip_self && instance == p->instance && optixGetPrimitiveIndex() == p->primitive))
    optixIgnoreIntersection();
}
extern "C" __global__ void __closesthit__query() {
  QueryPayload *p = (QueryPayload *)unpack_ptr(optixGetPayload_0(), optixGetPayload_1());
  SceneQuery &q = *p->result;
  q.hit = true;
  q.geometry = (HitGroupData *)optixGetSbtDataPointer();
  q.hit_primitive = optixGetPrimitiveIndex();
  q.distance = optixGetRayTmax();
  q.bary = q.geometry->curve_radii ? make_float2(optixGetCurveParameter(), 0) : optixGetTriangleBarycentrics();
}
extern "C" __global__ void __miss__query() {}

#if (VIBRT_SHADER_OPS & (7ull << 41))
extern "C" __device__ void __continuation_callable__shader(
    PrincipledGpu *m, const PathVertex *v, vibrt_cycles::float4 *out,
    unsigned int requested, int derivative) {
  evaluate_shader_vm_impl(m, *v, out, requested, derivative);
}
#endif

extern "C" __device__ void __continuation_callable__query(QueryCall *call) {
  call->result = query_scene_impl(*call->vertex, call->origin, call->direction,
                                 call->distance, call->only_local, call->skip_self,
                                 call->mask, call->tmin);
}

#if (VIBRT_SHADER_OPS & (7ull << 41))
extern "C" __device__ void __continuation_callable__material(MaterialCall *call) {
  call->count = eval_material_impl(*call->vertex, call->slots, call->weights,
                                   call->max_leaves, call->min_alpha, call->NoV);
}
extern "C" __device__ void __continuation_callable__emission(EmissionCall *call) {
  call->result = emission_alpha_impl(*call->vertex, call->include_emission);
}
extern "C" __device__ void __continuation_callable__shadow(ShadowCall *call) {
  call->result = surface_shadow_transparency_impl(*call->vertex);
}
#endif
