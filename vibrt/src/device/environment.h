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
  float fx = u * (float)w - 0.5f;
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
  x0 = wrap(x0, w);
  y0 = max(0, min(h - 1, y0));
  int y1 = max(0, min(h - 1, y0 + 1));
  auto fetch = [&](int x, int y) {
    const float *p = &data[(y * w + x) * 3];
    return make_float3(p[0], p[1], p[2]);
  };
  float3 c00 = fetch(x0, y0);
  float3 c10 = fetch(x1, y0);
  float3 c01 = fetch(x0, y1);
  float3 c11 = fetch(x1, y1);
  float3 c0 = c00 * (1.0f - dx) + c10 * dx;
  float3 c1 = c01 * (1.0f - dx) + c11 * dx;
  return c0 * (1.0f - dy) + c1 * dy;
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

static __device__ int cdf_search(const float *cdf, int n, float r) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (cdf[mid + 1] < r)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

struct EnvSample {
  float3 dir;
  float3 L;
  float pdf; // solid-angle PDF
};

// Power heuristic (β = 2) for two-strategy MIS.
static __forceinline__ __device__ float power_heuristic(float pa, float pb) {
  float a = pa * pa;
  float b = pb * pb;
  return a / fmaxf(a + b, 1e-20f);
}

// Solid-angle PDF of the envmap importance sampler for a given world-space dir.
// Inverse of sample_envmap: maps (theta, phi) back to the texel and reads the
// luminance, normalised by envmap_integral.
//
// Sampling distribution: each pixel (x, y) is picked with discrete probability
// P(x, y) = lum_raw × sin(θ_y) / Z, where Z = Σ lum_raw × sin(θ_y) =
// envmap_integral. The pixel covers solid-angle Ω = (2π² sin θ) / (W·H), so the
// solid-angle pdf is P / Ω = lum_raw × W·H / (2π² × Z) — no sin θ in the
// denominator, and `lum` must come from the raw texel (no world_strength) so
// the strength factor stays on the radiance side and direct-light contribution
// scales correctly with `world_strength`.
static __device__ float envmap_pdf(float3 dir) {
  // Active for both single-layer (world_type==1) and mixed (==2) envmaps.
  // The CDF lives on `envmap_data` either way (mixed worlds rasterise the
  // blend on the host so this PDF lookup stays consistent with the
  // direction sampler).
  if (params.world_type == 0 || params.envmap_integral <= 0.0f)
    return 0.0f;
  // For mixed worlds `envmap_rotation_z_rad` is 0 and this collapses to
  // identity — consistent with `world_background`'s shared lookup path.
  float c = cosf(-params.envmap_rotation_z_rad);
  float s = sinf(-params.envmap_rotation_z_rad);
  float3 d = make_float3(dir.x * c - dir.y * s, dir.x * s + dir.y * c, dir.z);
  float theta = acosf(fminf(fmaxf(d.z, -1.0f), 1.0f));
  float phi = atan2f(d.y, d.x);
  float u = phi * (0.5f * INV_PIf);
  if (u < 0.0f)
    u += 1.0f;
  float v = theta * INV_PIf;
  int w = params.envmap_width;
  int h = params.envmap_height;
  int x = min((int)(u * w), w - 1);
  int y = min((int)(v * h), h - 1);
  const float *px = &params.envmap_data[(y * w + x) * 3];
  float lum_raw =
      0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2];
  return lum_raw * (float)(w * h) /
         (2.0f * M_PIf * M_PIf * params.envmap_integral);
}

static __device__ EnvSample sample_envmap(RNG &rng) {
  EnvSample s;
  s.dir = make_float3(0, 0, 1);
  s.L = make_float3(0, 0, 0);
  s.pdf = 0.0f;
  // Active for both single-layer and mixed envmaps; the CDF lives on
  // `envmap_data` for both cases (host rasterises the mix for type==2).
  if (params.world_type == 0 || params.envmap_integral <= 0.0f)
    return s;
  int w = params.envmap_width;
  int h = params.envmap_height;
  float u1 = rng.next();
  float u2 = rng.next();
  int y = cdf_search(params.envmap_marginal_cdf, h, u1);
  const float *row = &params.envmap_conditional_cdf[y * (w + 1)];
  int x = cdf_search(row, w, u2);

  float v = ((float)y + 0.5f) / (float)h;
  float u = ((float)x + 0.5f) / (float)w;
  float theta = v * M_PIf;
  float phi = u * 2.0f * M_PIf;
  float sin_t = sinf(theta);
  float3 d = make_float3(sin_t * cosf(phi), sin_t * sinf(phi), cosf(theta));
  // Apply rotation (inverse of what `world_background` does). For mixed
  // worlds `envmap_rotation_z_rad` is left at 0 (the rasterised grid is
  // already in world space) so this collapses to the identity, which
  // is the correct "undo" of `world_background`'s identity lookup.
  float c = cosf(params.envmap_rotation_z_rad);
  float sr = sinf(params.envmap_rotation_z_rad);
  s.dir = make_float3(d.x * c - d.y * sr, d.x * sr + d.y * c, d.z);
  // Read s.L from the same rasterised grid the CDF / pdf were built
  // from. Returning a different (higher-res) radiance value for the
  // same direction would make `f / pdf` inconsistent and over-count
  // bright bands of the mixed envmap. The slight loss in HDRI peak
  // brightness on NEE samples is compensated by BSDF-sampled paths
  // (which DO read high-res via `world_background`) — MIS combines
  // both strategies into an unbiased estimator only when each
  // strategy's `f / pdf` is internally consistent.
  const float *px = &params.envmap_data[(y * w + x) * 3];
  s.L = make_float3(px[0], px[1], px[2]) * params.world_strength;

  // pdf is over raw texel luminance only — see envmap_pdf for the derivation.
  // Folding world_strength into `lum` here would cancel out of the
  // L/pdf ratio in NEE and make contribution invariant to `world_strength`.
  float lum_raw =
      0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2];
  if (params.envmap_integral > 0.0f) {
    s.pdf = lum_raw * (float)(w * h) /
            (2.0f * M_PIf * M_PIf * params.envmap_integral);
  }
  return s;
}
