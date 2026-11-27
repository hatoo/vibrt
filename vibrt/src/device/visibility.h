// ---------- Visibility (shadow rays) ----------
// Ray types: 0 = radiance, 1 = shadow, 2 = query. SBT stride = 3.
// Shadow hit group
// is a dummy (CH disabled, no AH) so only the miss program runs, which sets
// payload[0]=1 meaning "reached infinity = light visible".
static __device__ bool shadow_visible(float3 P, float3 dir, float tmax) {
  unsigned int miss_flag = 0;
  // Mask bit 1 = "blocks shadow rays"; instances with cast_shadow=false clear
  // it (e.g. classroom's paper-lantern drum) so NEE sees through them.
  optixTrace(params.traversable, P, dir, 0.0f, tmax, 0.0f, OptixVisibilityMask(0x02),
             OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 1, 3, 1,
             miss_flag);
  return miss_flag != 0;
}

// Same visibility query, but returns RGB beam transmittance through any
// volumes the shadow ray traverses on its way to the light. Volume-only
// boundaries are walked iteratively (CH fires on the radiance hit group,
// updates the local stack, and we re-trace from the offset surface).
//
// Returns (0, 0, 0) on opaque occlusion. Capped at `VOL_STACK_MAX * 4`
// trace iterations to bound the worst case (alternating boundary
// crossings); past that we conservatively assume the light is unreachable
// — pathological scenes only.
static __device__ __noinline__ float4 emission_alpha(const PathVertex &v, bool include_emission);
static __device__ __noinline__ float3 surface_shadow_transparency(const PathVertex &v);
static __device__ __noinline__ float3 shadow_volume_transmittance(
    float3 P, float3 dir, float tmax, VolumeStack stack) {
  float3 tr = make_float3(1, 1, 1);
  float3 cur_origin = P;
  float remaining = tmax;
  const unsigned int max_iters = VOL_STACK_MAX * 4 + params.max_transparent_bounces + 1;
  unsigned int transparent_bounces = 0;

  for (unsigned int iter = 0; iter < max_iters; iter++) {
    if (remaining <= 0.0f)
      return tr;

    // Trace using the radiance ray type so CH fires and we get hit info.
    PathVertex v;
    v.hit = 0;
    unsigned int hi, lo;
    pack_ptr(&v, hi, lo);
    optixTrace(params.traversable, cur_origin, dir, 0.0f, remaining, 0.0f,
               OptixVisibilityMask(0x02), OPTIX_RAY_FLAG_NONE, 0, 3, 0, hi, lo);

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

    // Volume boundaries update the medium without changing surface throughput.
    if (v.mat != nullptr && v.mat->volume != nullptr &&
        v.mat->volume_only != 0) {
      bool entering = dot3(v.Ng, dir) < 0.0f;
      if (entering)
        volume_stack_push(stack, v.mat->volume);
      else
        volume_stack_pop(stack, v.mat->volume);
      // Offset onto the side of the boundary we just entered.
      float3 next = offset_surface(v.P, v.Ng, dir);
      remaining -= t_seg + dot3(next - v.P, dir);
      cur_origin = next;
      continue;
    }
    if (transparent_bounces++ >= params.max_transparent_bounces)
      return make_float3(0, 0, 0);
    v.path_flags = 2u;
    tr = tr * surface_shadow_transparency(v);
    if (luminance(tr) < 1e-6f)
      return make_float3(0, 0, 0);
    if (v.mat && v.mat->volume) {
      if (dot3(v.Ng, dir) < 0) volume_stack_push(stack, v.mat->volume);
      else volume_stack_pop(stack, v.mat->volume);
    }
    float3 next = offset_surface(v.P, v.Ng, dir);
    remaining -= t_seg + dot3(next - v.P, dir);
    cur_origin = next;
  }
  // Exceeded the iteration cap — conservatively block.
  return make_float3(0, 0, 0);
}

static __device__ __forceinline__ float3 shadow_transmittance(float3 P, float3 dir, float tmax,
                                                              const VolumeStack &stack,
                                                              float3 Ng = make_float3(0, 0, 0)) {
  float3 start = dot3(Ng, Ng) > 0 ? offset_surface(P, Ng, dir) : P;
  if (tmax < 1e19f) {
    // Offset both endpoints, then aim at the adjusted endpoint. This also
    // works for tiny lights and scenes translated far from the origin.
    auto end = vibrt_cycles::ray_offset(cycles_vector(P + dir * tmax), cycles_vector(-dir));
    float3 delta = make_float3(end.x, end.y, end.z) - start;
    if (dot3(delta, dir) <= 0)
      return make_float3(0, 0, 0);
    tmax = sqrtf(dot3(delta, delta));
    dir = delta / tmax;
  }
  P = start;
  // Keep binary visibility outside the volume walk's call frame. Most shadow
  // rays need neither its private stack copy nor its boundary-hit storage.
  // Only scenes without transparent closures can use the binary query.
  if (!params.has_shadow_transparency && params.world_volume == nullptr && stack.depth == 0) {
    return shadow_visible(P, dir, tmax) ? make_float3(1, 1, 1)
                                      : make_float3(0, 0, 0);
  }
  // Copy only for the volume walk, which mutates its own boundary stack.
  return shadow_volume_transmittance(P, dir, tmax, stack);
}
