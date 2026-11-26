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
  return make_float3(expf(-v->sigma_t[0] * t), expf(-v->sigma_t[1] * t),
                     expf(-v->sigma_t[2] * t));
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
  float g2 = g * g;
  float denom = 1.0f + g2 - 2.0f * g * cos_theta;
  return (1.0f - g2) /
         (4.0f * M_PIf * sqrtf(fmaxf(denom, 1e-12f)) * fmaxf(denom, 1e-12f));
}

// Sample a scatter direction from HG phase, returned in the local frame
// {x = T_perp1, y = T_perp2, z = forward}. Caller transforms to world.
static __device__ float3 phase_hg_sample(float g, float u1, float u2,
                                         float &pdf) {
  float cos_theta;
  if (fabsf(g) < 1e-3f) {
    cos_theta = 1.0f - 2.0f * u1; // isotropic
  } else {
    float k = (1.0f - g * g) / (1.0f - g + 2.0f * g * u1);
    cos_theta = (1.0f + g * g - k * k) / (2.0f * g);
  }
  cos_theta = fmaxf(-1.0f, fminf(1.0f, cos_theta));
  float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
  float phi = 2.0f * M_PIf * u2;
  pdf = phase_hg_eval(g, cos_theta);
  return make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
}
