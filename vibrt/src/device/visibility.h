// ---------- Visibility (shadow rays) ----------
// Ray types: 0 = radiance, 1 = shadow. SBT stride = 2.
// Shadow hit group
// is a dummy (CH disabled, no AH) so only the miss program runs, which sets
// payload[0]=1 meaning "reached infinity = light visible".
static __device__ bool shadow_visible(float3 P, float3 dir, float tmax) {
  unsigned int miss_flag = 0;
  // Mask bit 1 = "blocks shadow rays"; instances with cast_shadow=false clear
  // it (e.g. classroom's paper-lantern drum) so NEE sees through them.
  optixTrace(params.traversable, P, dir, 1e-4f, tmax - 1e-3f, 0.0f,
             OptixVisibilityMask(0x02),
             OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                 OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
             1, 2, 1, miss_flag);
  return miss_flag != 0;
}

// Same visibility query, but returns RGB beam transmittance through any
// volumes the shadow ray traverses on its way to the light. Volume-only
// boundaries are walked iteratively (CH fires on the radiance hit group,
// updates the local stack, and we re-trace from the next epsilon).
//
// Returns (0, 0, 0) on opaque occlusion. Capped at `VOL_STACK_MAX * 4`
// trace iterations to bound the worst case (alternating boundary
// crossings); past that we conservatively assume the light is unreachable
// — pathological scenes only.
static __device__ __noinline__ float4 emission_alpha(const PathVertex &v, bool include_emission);
static __device__ __noinline__ float3 shadow_transmittance(float3 P, float3 dir, float tmax,
                                              VolumeStack stack) {
  // Fast path: no volumes anywhere. Standard binary visibility — the
  // any-hit handler already lets transmissive surfaces pass through
  // (`m->transmission > 0.5f -> optixIgnoreIntersection`), so direct
  // light reaches the pool floor through clear water without paying
  // for the iterative loop on every shadow ray.
  if (params.world_volume == nullptr && stack.depth == 0) {
    return shadow_visible(P, dir, tmax) ? make_float3(1, 1, 1)
                                        : make_float3(0, 0, 0);
  }

  float3 tr = make_float3(1, 1, 1);
  float3 cur_origin = P;
  float remaining = tmax;
  const int max_iters = VOL_STACK_MAX * 4 + 4;

  for (int iter = 0; iter < max_iters; iter++) {
    if (remaining <= 1e-4f)
      return tr;

    // Trace using the radiance ray type so CH fires and we get hit info.
    PathVertex v;
    v.hit = 0;
    unsigned int hi, lo;
    pack_ptr(&v, hi, lo);
    optixTrace(params.traversable, cur_origin, dir, 1e-4f, remaining - 1e-3f,
               0.0f, OptixVisibilityMask(0x02), OPTIX_RAY_FLAG_NONE, 0, 2, 0,
               hi, lo);

    const Volume *cur_vol = volume_stack_top(stack);
    float t_seg;
    if (v.hit == 0) {
      t_seg = remaining; // ran out before any geometry
    } else {
      float3 d = v.P - cur_origin;
      t_seg = sqrtf(dot3(d, d));
    }

    // Beam transmittance through the current volume on this segment.
    if (cur_vol != nullptr) {
      tr = tr * vol_beam_transmittance(cur_vol, t_seg);
      if (luminance(tr) < 1e-6f)
        return make_float3(0, 0, 0);
    }

    if (v.hit == 0)
      return tr; // reached light

    // Surface in the way. Three cases:
    //   1. Pure volume container — walk past, push/pop the volume stack.
    //   2. Transmissive surface (Glass / Refraction / `transmission > 0`) —
    //      attenuate by (1−F)·base_color and continue past, so the sun
    //      reaches the pool floor through the water plane. Without this
    //      the water acts as an opaque occluder for NEE and the only
    //      light hitting underwater surfaces is the path-traced indirect
    //      bounce, which is dim and noisy.
    //   3. Anything else — opaque, return zero.
    if (v.mat != nullptr && v.mat->volume != nullptr &&
        v.mat->volume_only != 0) {
      bool entering = dot3(v.Ng, dir) < 0.0f;
      if (entering)
        volume_stack_push(stack, v.mat->volume);
      else
        volume_stack_pop(stack, v.mat->volume);
      // Advance just past the surface. Use ray dir (not Ng) — we want to
      // stay on the same side of the boundary we just crossed.
      float p_scale =
          fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
      float eps = fmaxf(1e-4f, p_scale * 1e-5f);
      cur_origin = v.P + dir * eps;
      remaining -= t_seg + eps;
      continue;
    }
    if (v.mat != nullptr && v.mat->shader_has_alpha) {
      v.path_flags = 2u;
      float transparent = 1.0f - emission_alpha(v, false).w;
      if (transparent > 0.0f) {
        tr = tr * transparent;
        if (luminance(tr) < 1e-6f) return make_float3(0,0,0);
        float eps = fmaxf(1e-4f, fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)),fabsf(v.P.z))*1e-5f);
        cur_origin = v.P + dir * eps;
        remaining -= t_seg + eps;
        continue;
      }
    }
    if (v.mat != nullptr && v.mat->transmission > 0.0f) {
      // Transparent shadow: keep the ray going through the surface,
      // attenuated by `(1 − F) · base_color · transmission`. We can't
      // call `eval_material` here because it's defined later in the
      // translation unit and forward-declaring it just to share the
      // (textured) base colour would pull in a lot of unrelated state;
      // the constant `base_color` is correct for water / clear-glass
      // (the common case) and only loses tint texture variation, which
      // is rarely present on transmissive shaders.
      float3 base = make_f3(v.mat->base_color);
      float cos_i = fabsf(dot3(v.Ns, dir));
      float F0_d = ((v.mat->ior - 1.0f) / (v.mat->ior + 1.0f)) *
                   ((v.mat->ior - 1.0f) / (v.mat->ior + 1.0f))
                   * (2.0f * v.mat->specular_ior_level);
      float F = schlick_scalar(cos_i, F0_d);
      float3 t_factor = base * (1.0f - F) * v.mat->transmission;
      // Surfaces with `transmission < 1` (e.g. Principled at 0.5) still
      // partially absorb — fold the missing fraction into 0 so the ray
      // does NOT pretend to also transmit the diffuse / specular share.
      tr = tr * t_factor;
      if (luminance(tr) < 1e-6f)
        return make_float3(0, 0, 0);
      float p_scale =
          fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
      float eps = fmaxf(1e-4f, p_scale * 1e-5f);
      cur_origin = v.P + dir * eps;
      remaining -= t_seg + eps;
      continue;
    }
    return make_float3(0, 0, 0); // opaque occlusion
  }
  // Exceeded the iteration cap — conservatively block.
  return make_float3(0, 0, 0);
}
