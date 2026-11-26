extern "C" __global__ void __closesthit__ch() {
  fill_path_vertex(get_path_vertex());
}

extern "C" __global__ void __closesthit__shadow() {
  // Not used when TERMINATE_ON_FIRST_HIT is set — kept to satisfy SBT.
}

// Shared by radiance and shadow ray types: alpha-mask cutout, plus
// transparent-shadow pass-through for transmissive materials on shadow rays.
extern "C" __global__ void __anyhit__ah() {
  HitGroupData *hg = (HitGroupData *)optixGetSbtDataPointer();
  unsigned int prim = optixGetPrimitiveIndex();
  PrincipledGpu *m = hg->mat;
  if (hg->material_indices != nullptr && hg->num_materials > 0) {
    unsigned int mi = hg->material_indices[prim];
    if ((int)mi < hg->num_materials)
      m = hg->materials[mi];
  }
  // Shadow rays pass through transmissive surfaces (approximation of
  // Cycles' transparent shadows — no color attenuation for now).
  // Translucent thin sheets (Add(Diffuse, Translucent), sss_weight>0)
  // are NOT auto-passed: Cycles only lets ~30-50% through them, and
  // letting them be fully transparent makes the foreground pool floor
  // ~50% brighter than Cycles' reference. Cycles-style soft shadows
  // through translucent BSDFs would need a payload-accumulating
  // shadow-ray loop; for now respecting the alpha-cutout silhouette
  // (which is what's checked further below) gets us closer to
  // reference than full pass-through.
  //
  // NB: full pass-through over-delivers light entering through glass at
  // grazing angles (classroom exterior_fillLight ~2.4× over; clean
  // glass-pane test 2.58×). But BLOCKING glass shadows (the Cycles-exact
  // behaviour, delivering via refraction paths) under-delivers badly here
  // because vibrt's refraction paths under-sample at practical SPP
  // (flat_archiviz 1.00→0.92, junk_shop 1.07→0.73). The correct fix is
  // Fresnel-attenuated transparent shadows via an accumulating shadow-ray
  // loop; until then full pass-through is the better net approximation.
  bool is_shadow_ray =
      (optixGetRayFlags() & OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT) != 0;
  if (is_shadow_ray && m->transmission > 0.5f) {
    optixIgnoreIntersection();
  }
  // Cycles' Principled `Alpha < 1` mixes the BSDF with a Transparent
  // BSDF; shadow rays through the (1-alpha) portion pass straight
  // through and `alpha` of them are blocked by the opaque BSDF. We
  // approximate this stochastically: hash ray origin + direction +
  // primitive index into [0, 1) and ignore (pass through) when the
  // hash exceeds alpha_blend. Across many shadow rays through the
  // same surface the expected pass-through fraction is `(1-alpha_blend)`,
  // matching Cycles' transparent-branch fraction. PCG hash gives
  // per-ray decorrelation without needing to thread the path RNG
  // through OptiX's any-hit payload (which would require restructuring
  // every NEE call site). With the pure full-pass-through approximation
  // the camera-side window mix worked but rooms ended up ~5% too bright
  // because shadow rays through windows weren't attenuated.
  float shadow_alpha=m->shader_has_alpha?m->alpha_blend:1.0f;
  if(is_shadow_ray && m->shader_has_alpha && ((m->shader_outputs&(1u<<7))||m->secondary||m->left_subtree)) {
    PathVertex v;fill_path_vertex(&v);shadow_alpha=emission_alpha(v,false).w;
  }
  if (is_shadow_ray && shadow_alpha < 1.0f) {
    float3 ray_o = optixGetWorldRayOrigin();
    float3 ray_d = optixGetWorldRayDirection();
    unsigned int h = __float_as_uint(ray_o.x) ^ __float_as_uint(ray_o.y * 1.31f) ^
                     __float_as_uint(ray_o.z * 1.71f) ^
                     __float_as_uint(ray_d.x * 2.13f) ^
                     __float_as_uint(ray_d.y * 2.71f) ^
                     __float_as_uint(ray_d.z * 3.13f) ^
                     (prim * 747796405u + 2891336453u);
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    float u = (h >> 9) * (1.0f / 8388608.0f);
    if (u >= shadow_alpha) {
      optixIgnoreIntersection();
    }
  }
  // Pure volume-container surfaces are invisible to *binary* shadow rays
  // (the legacy `shadow_visible` path used by surface NEE on volumeless
  // scenes — TERMINATE_ON_FIRST_HIT is set there). Without this check the
  // smoke's mesh would shadow lights behind it as if it were opaque.
  //
  // We do NOT skip volume_only surfaces on radiance rays — `trace_path`
  // depends on closest-hit firing at the boundary so it can update the
  // volume stack. `shadow_transmittance` similarly uses the radiance ray
  // type (no terminate-on-first-hit) so CH runs there too.
  if (is_shadow_ray && m->volume != nullptr && m->volume_only != 0) {
    optixIgnoreIntersection();
  }
  if (m->alpha_threshold <= 0.0f || m->base_color_tex == nullptr)
    return;
  // Recompute UV from barycentrics.
  if (hg->uvs == nullptr)
    return;
  int i0 = hg->indices[prim * 3 + 0];
  int i1 = hg->indices[prim * 3 + 1];
  int i2 = hg->indices[prim * 3 + 2];
  float2 bary = optixGetTriangleBarycentrics();
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
  // Bilinear fetch of the alpha channel, wrapped.
  int w = m->base_color_tex_w;
  int h = m->base_color_tex_h;
  if (w <= 0 || h <= 0)
    return;
  float u = uv.x - floorf(uv.x);
  float v = uv.y - floorf(uv.y);
  float fx = u * (float)w - 0.5f;
  // Blender's image.pixels (and what the exporter writes to the bin) is
  // bottom-up: buffer row 0 is the bottom of the image. Blender UV v=0 is
  // also the bottom, so no flip on the y axis.
  float fy = v * (float)h - 0.5f;
  int x0 = (int)floorf(fx);
  int y0 = (int)floorf(fy);
  float dx = fx - (float)x0;
  float dy = fy - (float)y0;
  auto wrap = [](int v, int m) {
    int r = v % m;
    return r < 0 ? r + m : r;
  };
  int x1 = wrap(x0 + 1, w);
  int y1 = wrap(y0 + 1, h);
  x0 = wrap(x0, w);
  y0 = wrap(y0, h);
  const float *data = m->base_color_tex;
  int channels = m->base_color_tex_channels;
  auto fetch_alpha = [&](int x, int y) {
    return channels == 2 || channels == 4
               ? data[(y * w + x) * channels + channels - 1]
               : 1.0f;
  };
  float a00 = fetch_alpha(x0, y0);
  float a10 = fetch_alpha(x1, y0);
  float a01 = fetch_alpha(x0, y1);
  float a11 = fetch_alpha(x1, y1);
  float a0 = a00 * (1.0f - dx) + a10 * dx;
  float a1 = a01 * (1.0f - dx) + a11 * dx;
  float alpha = a0 * (1.0f - dy) + a1 * dy;
  if (alpha < m->alpha_threshold)
    optixIgnoreIntersection();
}

extern "C" __global__ void __miss__ms() {
  PathVertex *v = get_path_vertex();
  v->hit = 0;
}

extern "C" __global__ void __miss__shadow() { optixSetPayload_0(1u); }
