// ---------- Env map ----------
static __forceinline__ __device__ float3 rotate_y(float3 v, float c, float s) {
  return make_float3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

// Bilinear-filter an equirect texture (3-channel, packed) at a world-space
// direction `dir`. Wraps u (longitude is periodic) and clamps v (poles are
// not). `rotation` is a 3×3 row-major matrix applied to `dir` before the
// equirect lookup so per-layer Mapping rotations don't have to bake into
// pixels.
static __device__ float3 sample_equirect_layer(const float *data, int w, int h,
                                                const float *rotation,
                                                float3 dir) {
  float3 d = make_float3(rotation[0] * dir.x + rotation[1] * dir.y + rotation[2] * dir.z,
                         rotation[3] * dir.x + rotation[4] * dir.y + rotation[5] * dir.z,
                         rotation[6] * dir.x + rotation[7] * dir.y + rotation[8] * dir.z);
  float theta = acosf(fminf(fmaxf(d.z, -1.0f), 1.0f));
  float phi = atan2f(d.y, d.x);
  float u = phi * (0.5f * INV_PIf);
  if (u < 0.0f)
    u += 1.0f;
  float v = theta * INV_PIf;
  auto col = sample_image(data, w, h, 3, u, v, true);
  return make_float3(col.x, col.y, col.z);
}

// Build a 3×3 rotation matrix from `envmap_rotation_z_rad` for the legacy
// single-envmap path so it can share `sample_equirect_layer`.
static __forceinline__ __device__ void make_rotation_z(float angle,
                                                        float (&m)[9]) {
  float c = cosf(angle), s = sinf(angle);
  m[0] = c; m[1] = -s; m[2] = 0;
  m[3] = s; m[4] = c;  m[5] = 0;
  m[6] = 0; m[7] = 0;  m[8] = 1;
}

static __device__ float3 world_background(float3 dir, bool is_camera_ray) {
  if (params.world_type == 0) {
    return make_f3(params.world_color) * params.world_strength;
  }
  // Light-Path camera-ray split: camera-visible misses read layer b
  // directly (the artist's backplate), while all other rays fall through
  // to envmap_data (= the lighting envmap = layer a). The CDF / NEE
  // sampler stays on envmap_data so MIS pdfs match the radiance lighting
  // rays observe — camera rays don't go through MIS, so the asymmetry
  // is unbiased.
  if (params.world_type == 2 &&
      params.envmap_split_by_camera_ray != 0 &&
      is_camera_ray &&
      params.envmap_data_b != nullptr) {
    float3 col = sample_equirect_layer(params.envmap_data_b,
                                       params.envmap_width_b,
                                       params.envmap_height_b,
                                       params.envmap_rotation_b, dir);
    return col * params.envmap_strength_b;
  }
  // Both single-layer and mixed worlds read from `envmap_data` so that
  // MIS-combined NEE and BSDF-sampled-miss strategies see the same
  // radiance at any given direction (essential for unbiased estimation).
  // For mixed worlds, `envmap_data` is the host-rasterised mix at CDF
  // resolution; the per-layer high-res `envmap_data_a` / `_b` are
  // currently kept around for future high-res sampling once the CDF
  // construction also moves to the high-res grid.
  float rot[9];
  make_rotation_z(-params.envmap_rotation_z_rad, rot);
  float3 col = sample_equirect_layer(params.envmap_data,
                                     params.envmap_width,
                                     params.envmap_height, rot, dir);
  return col * params.world_strength;
}

struct EnvSample {
  float3 dir;
  float3 L;
  float pdf; // solid-angle PDF
};

// Preserve zero-density guards around the upstream MIS helper.
static __forceinline__ __device__ float power_heuristic(float pa, float pb) {
  return pa > 0 ? vibrt_cycles::power_heuristic(pa, pb) : 0;
}

static __device__ vibrt_background::Distribution env_distribution() {
  return {{params.envmap_width, params.envmap_height}, params.envmap_marginal_cdf,
          params.envmap_conditional_cdf};
}

static __device__ float envmap_pdf(float3 dir) {
  if (params.world_type == 0 || params.envmap_integral <= 0)
    return 0;
  float c = cosf(-params.envmap_rotation_z_rad), s = sinf(-params.envmap_rotation_z_rad);
  auto d = vibrt_cycles::make_float3(dir.x * c - dir.y * s, dir.x * s + dir.y * c, dir.z);
  auto distribution = env_distribution();
  return vibrt_background::background_map_pdf(&distribution, d);
}

static __device__ EnvSample sample_envmap(RNG &rng) {
  EnvSample sample{make_float3(0, 0, 1), make_float3(0, 0, 0), 0};
  if (params.world_type == 0 || params.envmap_integral <= 0)
    return sample;
  auto distribution = env_distribution();
  // Exclude exactly zero: an initial run of empty CDF bins has no invertible density.
  float u = fmaxf(rng.next(), 0x1p-24f), v = fmaxf(rng.next(), 0x1p-24f);
  auto d = vibrt_background::background_map_sample(
      &distribution, vibrt_cycles::make_float2(u, v), &sample.pdf);
  float c = cosf(params.envmap_rotation_z_rad), s = sinf(params.envmap_rotation_z_rad);
  sample.dir = make_float3(d.x * c - d.y * s, d.x * s + d.y * c, d.z);
  sample.L = world_background(sample.dir, false);
  return sample;
}
