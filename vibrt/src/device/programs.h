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
