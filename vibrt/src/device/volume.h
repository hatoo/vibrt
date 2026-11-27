// ---------- Homogeneous volumes and phase functions ----------

// Maximum nesting depth for mesh-bounded volumes within a single ray. Four is
// generous: real scenes top out at "two overlapping volumes" (e.g. a fog cube
// inside a smoke cube). World volume sits implicitly below the stack.
#define VOL_STACK_MAX 4

struct VolumeStack {
  const Volume *entries[VOL_STACK_MAX];
  int depth;
};

static __forceinline__ __device__ const Volume *
volume_stack_top(const VolumeStack &s) {
  if (s.depth > 0)
    return s.entries[s.depth - 1];
  return params.world_volume; // may be nullptr (vacuum)
}

static __forceinline__ __device__ void
volume_stack_push(VolumeStack &s, const Volume *v) {
  if (v == nullptr)
    return;
  if (s.depth < VOL_STACK_MAX)
    s.entries[s.depth++] = v;
  // Overflow silently — losing the deepest nesting is preferable to corrupt
  // memory; the scene would already be pathological at this nesting depth.
}

static __forceinline__ __device__ void
volume_stack_pop(VolumeStack &s, const Volume *v) {
  // Pop by reference equality scanned from top. BVH order isn't strict LIFO
  // (a back-face exit can come before the matching front-face entry was
  // recorded if rays miss tessellation), so a strict pop would desync.
  for (int i = s.depth - 1; i >= 0; i--) {
    if (s.entries[i] == v) {
      for (int j = i; j < s.depth - 1; j++)
        s.entries[j] = s.entries[j + 1];
      s.depth--;
      return;
    }
  }
}

// Average per-channel σ_t — used as the scalar pdf for distance sampling.
// We then track per-channel beam transmittance separately so chromatic
// volumes don't lose colour information through the sampler.
static __forceinline__ __device__ float vol_sigma_t_avg(const Volume *v) {
  return (v->sigma_t[0] + v->sigma_t[1] + v->sigma_t[2]) * (1.0f / 3.0f);
}

static __forceinline__ __device__ float3 vol_beam_transmittance(const Volume *v,
                                                                float t) {
  auto tr = vibrt_cycles::volume_color_transmittance(cycles_vector(make_f3(v->sigma_t)), t);
  return make_float3(tr.x, tr.y, tr.z);
}

// Closed-form path emission integral over [0, t] for a homogeneous volume:
//   ∫_0^t exp(-σ_t s) σ_e ds = σ_e/σ_t · (1 - exp(-σ_t t))   (per channel)
// Computed deterministically — no MC noise from emissive media. Falls back
// to σ_e · t in the σ_t→0 limit (Beer-Lambert is identity, integrand = σ_e).
static __device__ float3 volume_emission(const Volume *v, unsigned int flags);
static __device__ float3 vol_path_emission(const Volume *v, float t, unsigned int flags) {
  float3 e = volume_emission(v, flags);
  if (e.x <= 0.0f && e.y <= 0.0f && e.z <= 0.0f)
    return make_float3(0, 0, 0);
  auto component = [t](float sig_t, float em) {
    if (em <= 0.0f)
      return 0.0f;
    if (sig_t < 1e-8f)
      return em * t;
    return em * (1.0f - expf(-sig_t * t)) / sig_t;
  };
  return make_float3(component(v->sigma_t[0], e.x),
                     component(v->sigma_t[1], e.y),
                     component(v->sigma_t[2], e.z));
}

// Henyey-Greenstein phase function p(cos θ) and importance sampling.
// p(μ) = (1 - g²) / (4π · (1 + g² - 2g·μ)^(3/2))
static __forceinline__ __device__ float phase_hg_eval(float g, float cos_theta) {
  return vibrt_cycles::phase_henyey_greenstein(cos_theta, g);
}

// Sample a scatter direction from HG phase, returned in the local frame
// {x = T_perp1, y = T_perp2, z = forward}. Caller transforms to world.
static __device__ float3 phase_hg_sample(float g, float u1, float u2,
                                         float &pdf) {
  auto d = vibrt_cycles::phase_henyey_greenstein_sample(
      vibrt_cycles::make_float3(0, 0, 1), g, vibrt_cycles::make_float2(u1, u2), &pdf);
  return make_float3(d.x, d.y, d.z);
}

// Select a finite emitter as a distance proposal. Conditioning both the sample
// and its density on the same independently chosen emitter avoids a costly sum
// over lights. The exponential component retains support everywhere.
static __device__ bool volume_equiangular_coefficients(float3 origin, float3 dir, float length,
                                                       RNG &rng, vibrt_volume::EquiangularCoefficients &coeffs) {
  int count = params.num_point_lights + params.num_spot_lights + params.num_rect_lights + params.num_mesh_lights;
  if (count == 0 || length <= 0 || length >= 1e19f)
    return false;
  int index = min((int)(rng.next() * count), count - 1);
  float3 P;
  if (index < params.num_point_lights)
    P = make_f3(params.point_lights[index].position);
  else if ((index -= params.num_point_lights) < params.num_spot_lights)
    P = make_f3(params.spot_lights[index].position);
  else if ((index -= params.num_spot_lights) < params.num_rect_lights) {
    const auto &light = params.rect_lights[index];
    auto uv = light.disk ? vibrt_cycles::sample_uniform_disk(vibrt_cycles::make_float2(rng.next(), rng.next()))
                        : vibrt_cycles::make_float2(2 * rng.next() - 1, 2 * rng.next() - 1);
    P = make_f3(light.corner) + make_f3(light.u_axis) * (light.size_u * (0.5f + 0.5f * uv.x)) +
        make_f3(light.v_axis) * (light.size_v * (0.5f + 0.5f * uv.y));
  }
  else {
    index -= params.num_rect_lights;
    const auto &light = params.mesh_lights[index];
    if (!light.num_triangles)
      return false;
    int triangle = cdf_search(light.cdf, light.num_triangles, rng.next());
    const auto &t = light.triangles[triangle];
    float u = sqrtf(rng.next()), v = rng.next();
    P = make_f3(t.v0) * (1 - u) + make_f3(t.v1) * (u * (1 - v)) + make_f3(t.v2) * (u * v);
  }
  coeffs = {cycles_vector(P), {0, length}};
  float projection = dot3(P - origin, dir);
  float3 perpendicular = P - origin - dir * projection;
  float D = sqrtf(dot3(perpendicular, perpendicular));
  return D > 1e-6f && atan2f(length - projection, D) - atan2f(-projection, D) > 1e-6f;
}
