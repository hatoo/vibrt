// ---------- IES profile lookup ----------
//
// Returns the IES table's normalised intensity multiplier in [0, 1] for
// the world-space sample direction `dir_world` (light → surface). When
// `ies_data == nullptr` returns 1.0 (no IES, isotropic).
//
// Layout of `ies_data` matches `upload_ies_buffers` in render.rs:
//   [0           .. n_v          ): vertical angles in degrees (theta)
//   [n_v         .. n_v + n_h    ): horizontal angles in degrees (phi)
//   [n_v + n_h   .. + n_v * n_h  ): normalised candelas, row-major over
//                                    phi × theta — `[h * n_v + v]`
//
// Convention: `light_rotation` is the light's object→world 3×3 (row-
// major). We invert (transpose for pure rotations) to convert
// `dir_world` into the light's local frame, then derive theta from the
// local -Z axis (the IES "down" convention) and phi from atan2 around
// local +Z.
static __device__ float ies_lookup(const float *ies_data,
                                   unsigned int n_v, unsigned int n_h,
                                   const float *light_rotation,
                                   float3 dir_world) {
  if (ies_data == nullptr || n_v == 0 || n_h == 0)
    return 1.0f;
  // light_rotation row-major: rows are world-space basis vectors of the
  // light's local axes. world→local = transpose(rotation), so the local
  // dir is (rotation[col 0] · world, rotation[col 1] · world, ...)
  // expressed as: local.x = R[0]·d + R[3]·d.y + R[6]·d.z (column 0).
  float3 local;
  local.x = light_rotation[0] * dir_world.x + light_rotation[3] * dir_world.y +
            light_rotation[6] * dir_world.z;
  local.y = light_rotation[1] * dir_world.x + light_rotation[4] * dir_world.y +
            light_rotation[7] * dir_world.z;
  local.z = light_rotation[2] * dir_world.x + light_rotation[5] * dir_world.y +
            light_rotation[8] * dir_world.z;
  float len = sqrtf(dot3(local, local));
  if (len < 1e-12f)
    return 1.0f;
  // theta from -Z (IES "down" convention). cos(theta) = dot(local, -Z) =
  // -local.z / len.
  float cos_t = fminf(1.0f, fmaxf(-1.0f, -local.z / len));
  float theta_deg = acosf(cos_t) * (180.0f / M_PIf);
  // phi: horizontal angle around the IES axis. Match Cycles'
  // `kernel/svm/ies.h`: `h_angle = atan2f(vector.x, vector.y) + PI`
  // (NOT atan2(y,x)) — the LM-63 Type-C horizontal convention. For
  // radially-symmetric profiles (n_h==1) phi is ignored, but directional
  // profiles (e.g. comet.ies, n_h=5) read the wrong lobe under the old
  // atan2(y,x) form, under-delivering the point/spot IES lamps.
  float phi_rad = atan2f(local.x, local.y) + M_PIf;
  float phi_deg = phi_rad * (180.0f / M_PIf);

  // Bilinear lookup in the table.
  const float *thetas = ies_data;
  const float *phis = ies_data + n_v;
  const float *cands = ies_data + n_v + n_h;

  // Vertical-angle bracket. Clamp to table range; IES profiles often
  // have a sparse range like [0, 90] (downlights) so anything below
  // theta[0] takes theta[0] and anything above theta[n_v-1] takes the
  // last entry (typically 0 candela for properly authored downlights).
  unsigned int v_lo = 0, v_hi = 0;
  float tv = 0.0f;
  if (theta_deg <= thetas[0]) {
    v_lo = 0; v_hi = 0;
  } else if (theta_deg >= thetas[n_v - 1]) {
    v_lo = n_v - 1; v_hi = n_v - 1;
  } else {
    for (unsigned int i = 0; i + 1 < n_v; i++) {
      if (thetas[i] <= theta_deg && theta_deg < thetas[i + 1]) {
        v_lo = i;
        v_hi = i + 1;
        float span = thetas[v_hi] - thetas[v_lo];
        tv = (span > 0.0f) ? (theta_deg - thetas[v_lo]) / span : 0.0f;
        break;
      }
    }
  }

  // Horizontal-angle bracket. n_h == 1 means radially symmetric — phi
  // is ignored. Otherwise fold phi according to the table's coverage:
  // phi_max ≤ 90 → quadrant symmetric, ≤ 180 → bilateral, else full.
  unsigned int h_lo = 0, h_hi = 0;
  float th = 0.0f;
  if (n_h > 1) {
    float phi_max = phis[n_h - 1];
    float p = phi_deg;
    if (phi_max <= 90.0f) {
      p = fmodf(p, 180.0f);
      if (p > 90.0f) p = 180.0f - p;
      if (p > 90.0f) p = 90.0f;
    } else if (phi_max <= 180.0f) {
      if (p > 180.0f) p = 360.0f - p;
    }
    if (p <= phis[0]) {
      h_lo = 0; h_hi = 0;
    } else if (p >= phis[n_h - 1]) {
      h_lo = n_h - 1; h_hi = n_h - 1;
    } else {
      for (unsigned int i = 0; i + 1 < n_h; i++) {
        if (phis[i] <= p && p < phis[i + 1]) {
          h_lo = i;
          h_hi = i + 1;
          float span = phis[h_hi] - phis[h_lo];
          th = (span > 0.0f) ? (p - phis[h_lo]) / span : 0.0f;
          break;
        }
      }
    }
  }

  float a = cands[h_lo * n_v + v_lo];
  float b = cands[h_lo * n_v + v_hi];
  float c = cands[h_hi * n_v + v_lo];
  float d = cands[h_hi * n_v + v_hi];
  float ab = a * (1.0f - tv) + b * tv;
  float cd = c * (1.0f - tv) + d * tv;
  return ab * (1.0f - th) + cd * th;
}

// Sum of all sun-light NEE pdfs at a given world-space direction. Used
// to extend the MIS denominator across NEE strategies so envmap NEE,
// sun NEE, and BSDF-sampled paths combine into a single unbiased
// estimator. Returns 0 when `dir` is outside every sun's cone.
static __device__ float sum_sun_nee_pdf(float3 dir) {
  float p = 0.0f;
  for (int i = 0; i < params.num_sun_lights; i++) {
    SunLight &sl = params.sun_lights[i];
    float3 sd = make_f3(sl.direction);
    float cos_to_sun = dot3(dir, sd);
    if (cos_to_sun > sl.cos_angle) {
      float sun_omega = 2.0f * M_PIf * fmaxf(1.0f - sl.cos_angle, 1e-30f);
      p += 1.0f / sun_omega;
    }
  }
  return p;
}

// Envmap NEE pdf at a world-space direction (0 when there's no envmap
// or the bake's integral is empty).
static __device__ float sum_env_nee_pdf(float3 dir) {
  if (params.world_type != 0 && params.envmap_integral > 0.0f) {
    return envmap_pdf(dir);
  }
  return 0.0f;
}

// 3-strategy power-heuristic with β=2. Picks `pdf_self` against the
// other two strategies' pdfs so the chosen-sample's MIS weight forms a
// proper mixture density across env NEE / sun NEE / BSDF sampling.
static __device__ float power_heuristic_3(float pdf_self,
                                           float pdf_o1,
                                           float pdf_o2) {
  float a = pdf_self * pdf_self;
  float b = pdf_o1 * pdf_o1;
  float c = pdf_o2 * pdf_o2;
  float denom = a + b + c;
  return (denom > 0.0f) ? a / denom : 0.0f;
}

// Rectangular lights are selected at each shading point, rather than by
// global wattage: a distant, large panel must not starve nearby small lights.
// This is a positive importance estimate, not a visibility/BSDF test. The
// normal floor preserves support for transmission and mixed/bumped closures.
static __device__ float rect_light_importance(const AreaRectLight &ar,
                                              float3 P, float3 N) {
  float area = ar.size_u * ar.size_v * (ar.disk ? M_PIf * 0.25f : 1.0f);
  float brightness = fmaxf(luminance(make_f3(ar.emission)), 0.0f);
  if (area <= 0 || brightness <= 0)
    return 0;
  float3 delta = make_f3(ar.corner) + make_f3(ar.u_axis) * (0.5f * ar.size_u) +
                 make_f3(ar.v_axis) * (0.5f * ar.size_v) - P;
  float plane = -dot3(make_f3(ar.normal), delta);
  if (ar.two_sided == 0u && plane <= 0)
    return 0;
  float d2 = fmaxf(dot3(delta, delta), 1e-20f);
  float inv_d = rsqrtf(d2);
  float cosine = fabsf(plane) * inv_d;
  // Finite-area regularization limits the near-field estimate to 2*pi.
  float omega = area * cosine / (d2 + area * (0.5f * INV_PIf));
  float normal_factor = dot3(N, N) > 0 ?
      0.1f + 0.9f * fabsf(dot3(N, delta)) * inv_d : 1.0f;
  return brightness * omega * normal_factor;
}

static __device__ int sample_rect_light(float3 P, float3 N, float u, float &pmf) {
  pmf = 0;
  float total = 0;
  for (int i = 0; i < params.num_rect_lights; ++i)
    total += rect_light_importance(params.rect_lights[i], P, N);
  if (total <= 0)
    return -1;
  float target = u * total;
  int selected = -1;
  float weight = 0;
  for (int i = 0; i < params.num_rect_lights; ++i) {
    float w = rect_light_importance(params.rect_lights[i], P, N);
    if (w <= 0)
      continue;
    selected = i;
    weight = w;
    target -= w;
    if (target < 0)
      break;
  }
  pmf = weight / total;
  return selected;
}

static __device__ float rect_light_pmf(int index, float3 P, float3 N) {
  float total = 0;
  for (int i = 0; i < params.num_rect_lights; ++i)
    total += rect_light_importance(params.rect_lights[i], P, N);
  return total > 0 ? rect_light_importance(params.rect_lights[index], P, N) / total : 0;
}

static __device__ vibrt_rect::Distribution rect_distribution(const AreaRectLight &ar,
                                                             float3 P) {
  float3 corner = make_f3(ar.corner) - P;
  return vibrt_rect::prepare(dot3(corner, make_f3(ar.u_axis)),
                             dot3(corner, make_f3(ar.v_axis)),
                             dot3(corner, make_f3(ar.normal)), ar.size_u, ar.size_v);
}

static __device__ float3 sample_rect_position(const AreaRectLight &ar, float3 P,
                                               float u, float v, float &pdf) {
  if (ar.disk) {
    float r = sqrtf(u), phi = 2 * M_PIf * v;
    float3 point = make_f3(ar.corner) + make_f3(ar.u_axis) * (ar.size_u * (0.5f + 0.5f * r * cosf(phi))) +
                   make_f3(ar.v_axis) * (ar.size_v * (0.5f + 0.5f * r * sinf(phi)));
    float3 delta = point - P;
    float d2 = dot3(delta, delta);
    float cosine = fabsf(dot3(normalize3(delta), make_f3(ar.normal)));
    pdf = d2 / fmaxf(ar.size_u * ar.size_v * M_PIf * 0.25f * cosine, 1e-30f);
    return point;
  }
  auto distribution = rect_distribution(ar, P);
  float x, y;
  vibrt_rect::sample(distribution, u, v, x, y);
  pdf = vibrt_rect::pdf(distribution, x, y);
  return make_f3(ar.corner) + make_f3(ar.u_axis) * (x - distribution.x0) +
         make_f3(ar.v_axis) * (y - distribution.y0);
}

static __device__ float rect_direction_pdf(const AreaRectLight &ar, float3 P,
                                            float3 light_position) {
  if (ar.disk) {
    float3 delta = light_position - P;
    float cosine = fabsf(dot3(normalize3(delta), make_f3(ar.normal)));
    return dot3(delta, delta) / fmaxf(ar.size_u * ar.size_v * M_PIf * 0.25f * cosine, 1e-30f);
  }
  auto distribution = rect_distribution(ar, P);
  float3 delta = light_position - P;
  return vibrt_rect::pdf(distribution, dot3(delta, make_f3(ar.u_axis)),
                         dot3(delta, make_f3(ar.v_axis)));
}

// ---------- Direct lighting (NEE) ----------
//
// `vstack` is the shadow ray's starting volume stack. For surface NEE this is
// the same stack `trace_path` is currently maintaining; for in-scatter volume
// NEE the caller does the same thing (the scatter point is *inside* the
// current volume, not on a boundary, so the stack starts unchanged).
//
// `shadow_transmittance` returns RGB so each light contribution is attenuated
// by the per-channel volume transmittance. On scenes without volumes it
// short-circuits to the legacy binary `shadow_visible` path.
static __device__ __noinline__ float3 direct_light(const MaterialEvalGroup &e, float3 P,
                                      float3 wo, RNG &rng,
                                      const VolumeStack &vstack, bool rect_mis,
                                      SurfaceClosureCache &cache) {
  float3 L = make_float3(0, 0, 0);

  // Point lights: sample the finite sphere/disk, or its delta limit.
  for (int i = 0; i < params.num_point_lights; i++) {
    PointLight &pl = params.point_lights[i];
    float3 Pl, wi;
    float d;
    float light_weight = sample_point_position(make_f3(pl.position), pl.radius,
                                               pl.sphere, P, rng, Pl, wi, d);
    if (d < 1e-4f || light_weight <= 0)
      continue;
    float3 vis = shadow_transmittance(P, wi, d, vstack);
    if (luminance(vis) <= 0.0f)
      continue;
    BsdfEval b = eval_bsdf(e, wo, wi, cache);
    // IES: directional intensity multiplier in [0, 1] sampled in the
    // light's local frame. `wi` points from surface to light, so the
    // direction "from light toward surface" is `-wi`. ies_lookup
    // returns 1.0 when no IES is attached.
    float ies = ies_lookup(pl.ies_data, pl.ies_n_v, pl.ies_n_h,
                           pl.light_rotation, -wi);
    L = L + b.f * vis * point_light_emission(pl.shader, make_f3(pl.emission), Pl, -wi, d) * ies * light_weight;
  }

  // Sun lights: sample within cone, MIS-weight against BSDF sampling.
  // The kernel's `emission` field already encodes `Le_per_dir × sun_omega`
  // (= irradiance), so the unweighted estimator `bsdf × cos × emission`
  // equals `bsdf × cos × Le × sun_omega = f(x) / pdf` for a uniform-cone
  // sampler with `pdf = 1/sun_omega`. Multiplying by `w_sun` keeps the
  // estimator unbiased while letting the matching BSDF-side branch in
  // `trace_path`'s envmap-miss block carry the BSDF-sampled contribution
  // when a scatter ray happens to land inside the sun cone.
  for (int i = 0; i < params.num_sun_lights; i++) {
    SunLight &sl = params.sun_lights[i];
    float3 dir = make_f3(sl.direction);
    // Small cone sample around the sun direction.
    float cos_a = sl.cos_angle;
    float u1 = rng.next();
    float u2 = rng.next();
    float ct = 1.0f - u1 * (1.0f - cos_a);
    float st = sqrtf(fmaxf(0.0f, 1.0f - ct * ct));
    float phi = 2.0f * M_PIf * u2;
    float3 T, B;
    build_frame(dir, T, B);
    float3 wi =
        normalize3(T * (st * cosf(phi)) + B * (st * sinf(phi)) + dir * ct);
    float3 vis = shadow_transmittance(P, wi, 1e20f, vstack);
    if (luminance(vis) <= 0.0f)
      continue;
    BsdfEval b = eval_bsdf(e, wo, wi, cache);
    float sun_omega = 2.0f * M_PIf * fmaxf(1.0f - cos_a, 1e-30f);
    float pdf_sun = 1.0f / sun_omega;
    // 3-strategy MIS: balance sun NEE against the (residual) envmap NEE
    // and the BSDF sample. Without the env NEE term the denominator
    // missed the case where the envmap CDF picks a direction inside the
    // sun cone — rare but not zero, since the bake's clipped sun pixels
    // have nonzero luminance after `_extract_sun_from_bake_inplace`.
    float pdf_env_at_dir = sum_env_nee_pdf(wi);
    float w_sun = power_heuristic_3(pdf_sun, pdf_env_at_dir, b.pdf);
    L = L + b.f * vis * make_f3(sl.emission) * w_sun;
  }

  // Spot lights
  for (int i = 0; i < params.num_spot_lights; i++) {
    SpotLight &sp = params.spot_lights[i];
    float3 Pl, wi;
    float d;
    float light_weight = sample_point_position(make_f3(sp.position), sp.radius,
                                               sp.sphere, P, rng, Pl, wi, d);
    if (d < 1e-4f || light_weight <= 0)
      continue;
    // Direction from light to point = -wi, cos against spot axis
    float3 center_delta = make_f3(sp.position) - P;
    float cos_a = sp.sphere && dot3(center_delta, center_delta) < sp.radius * sp.radius
                      ? 1.0f : dot3(make_f3(sp.direction), -wi);
    if (cos_a <= sp.cos_outer)
      continue;
    float falloff = 1.0f;
    if (cos_a < sp.cos_inner) {
      float t =
          (cos_a - sp.cos_outer) / fmaxf(sp.cos_inner - sp.cos_outer, 1e-4f);
      falloff = t * t * (3.0f - 2.0f * t);
    }
    float3 vis = shadow_transmittance(P, wi, d, vstack);
    if (luminance(vis) <= 0.0f)
      continue;
    BsdfEval b = eval_bsdf(e, wo, wi, cache);
    float ies = ies_lookup(sp.ies_data, sp.ies_n_v, sp.ies_n_h,
                           sp.light_rotation, -wi);
    L = L + b.f * vis * point_light_emission(sp.shader, make_f3(sp.emission), Pl, -wi, d) * falloff * ies * light_weight;
  }

  // Area rect lights: point-dependent selection and solid-angle sampling.
  if (params.num_rect_lights > 0) {
    float pmf_light;
    int idx = sample_rect_light(P, e.slots[0].Ns, rng.next(), pmf_light);
    if (idx >= 0 && pmf_light > 0.0f) {
      AreaRectLight &ar = params.rect_lights[idx];
      float su = rng.next();
      float sv = rng.next();
      float pdf_solid;
      float3 Pl = sample_rect_position(ar, P, su, sv, pdf_solid);
      float3 ld = Pl - P;
      float d2 = dot3(ld, ld);
      float d = sqrtf(d2);
      if (d > 1e-4f) {
        float3 wi = ld / d;
        float cos_light = -dot3(make_f3(ar.normal), wi);
        // Two-sided rects emit from both faces — take |cos_light| so
        // surface points on either side of the panel see it.
        if (ar.two_sided != 0u)
          cos_light = fabsf(cos_light);
        if (cos_light > 0.0f) {
          float3 vis = shadow_transmittance(P, wi, d, vstack);
          if (luminance(vis) > 0.0f) {
            BsdfEval b = eval_bsdf(e, wo, wi, cache);
            float pdf = pdf_solid * pmf_light;
            if (pdf > 0.0f) {
              // Area rects emit along local +Z, but `ies_lookup` measures
              // theta from local -Z (the point/spot "down" convention). The
              // outgoing direction is -wi ≈ +Z, so pass `wi` (= -outgoing)
              // to map the emission axis onto the profile's theta=0 peak.
              // With -wi the lookup landed at theta≈180° (the profile's zero
              // backside) and the IES area light delivered nothing.
              float ies = ies_lookup(ar.ies_data, ar.ies_n_v, ar.ies_n_h,
                                     ar.light_rotation, wi);
              // MIS-weight against BSDF sampling — but ONLY for camera-
              // visible rects, since `intersect_rect_lights` (and thus the
              // BSDF-side emission branch in trace_path) skips hidden rects.
              // Hidden emissive-mesh rects are NEE-only by design; weighting
              // them would drop energy with nothing to compensate. For a
              // visible rect a sharp glossy lobe's reflection is delivered by
              // the BSDF-side branch, which NEE can't importance-sample.
              // At the path-length limit no BSDF ray will be traced, so
              // NEE is the only strategy and must retain its full weight.
              float w_light = (ar.camera_visible != 0u && rect_mis)
                                  ? power_heuristic(pdf, b.pdf)
                                  : 1.0f;
              L = L + b.f * vis * area_light_emission(ar, Pl, -wi, d) * ies * (w_light / pdf);
            }
          }
        }
      }
    }
  }

  // Mesh lights: two-stage CDF — outer over per-mesh-light total power,
  // inner over per-triangle area × luminance(emission). Same solid-angle
  // pdf math as area_rect (uniform sample on triangle, then the standard
  // d² / (area × cos_light) Jacobian). MIS-weighted against BSDF
  // sampling via `power_heuristic` — the matching closest-hit branch
  // in `trace_path`'s emission gate adds the BSDF-side weighted
  // contribution when a sampled wi happens to land on a mesh-light
  // triangle. Shadow ray uses standard `shadow_transmittance` which
  // subtracts ~1mm from tmax internally, so it doesn't self-intersect
  // the sampled light triangle.
  if (params.num_mesh_lights > 0) {
    float u = rng.next();
    int mli = cdf_search(params.mesh_light_cdf, params.num_mesh_lights, u);
    MeshLight &ml = params.mesh_lights[mli];
    float pmf_outer =
        (params.mesh_light_cdf[mli + 1] - params.mesh_light_cdf[mli]);
    if (pmf_outer > 0.0f && ml.num_triangles > 0) {
      float ut = rng.next();
      int ti = cdf_search(ml.cdf, (int)ml.num_triangles, ut);
      MeshLightTri &tri = ml.triangles[ti];
      float pmf_tri = ml.cdf[ti + 1] - ml.cdf[ti];
      if (pmf_tri > 0.0f) {
        // Uniform area sample on the triangle via the Turk √r mapping.
        float r1 = rng.next();
        float r2 = rng.next();
        float sq = sqrtf(r1);
        float b0 = 1.0f - sq;
        float b1 = sq * (1.0f - r2);
        float b2 = sq * r2;
        float3 v0 = make_f3(tri.v0);
        float3 v1 = make_f3(tri.v1);
        float3 v2 = make_f3(tri.v2);
        float3 Pl = v0 * b0 + v1 * b1 + v2 * b2;
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;
        float3 ng = cross3(e1, e2);
        float ng_len2 = dot3(ng, ng);
        if (ng_len2 > 1e-20f) {
          float3 n = ng * (1.0f / sqrtf(ng_len2));
          float3 ld = Pl - P;
          float d2 = dot3(ld, ld);
          float d = sqrtf(d2);
          if (d > 1e-4f) {
            float3 wi = ld / d;
            float cos_light = -dot3(n, wi);
            if (ml.two_sided != 0u)
              cos_light = fabsf(cos_light);
            if (cos_light > 0.0f) {
              float3 vis = shadow_transmittance(P, wi, d, vstack);
              if (luminance(vis) > 0.0f) {
                BsdfEval b = eval_bsdf(e, wo, wi, cache);
                float area = tri.area;
                float pdf_area = 1.0f / fmaxf(area, 1e-12f);
                float pdf_solid = pdf_area * d2 / cos_light;
                float pdf_light = pdf_solid * pmf_outer * pmf_tri;
                if (pdf_light > 0.0f) {
                  // Per-sample emission radiance. The constant `tri.emission`
                  // is the material's emission strength multiplier (the
                  // Cycles Principled `Emission` socket × `Emission
                  // Strength`). When `emission_tex` is set, sample the
                  // texture at the barycentric → UV-interpolated coordinate
                  // so a textured emissive surface (LED panel, CRT screen,
                  // billboard) delivers per-pixel detail at the NEE-sampled
                  // point — without this, all triangles of a textured
                  // emissive mesh deliver a uniform glow regardless of
                  // which texel they cover.
                  float3 Le = mesh_light_emission(tri,b1,b2,-wi,d);
                  // MIS against BSDF sampling. Power heuristic with β=2 —
                  // matches the envmap NEE block's choice. The matching
                  // BSDF-side branch in `trace_path` uses (1-w_light) on
                  // emission contributions that hit mesh-light triangles.
                  float w_light = power_heuristic(pdf_light, b.pdf);
                  L = L + b.f * vis * Le * (w_light / pdf_light);
                }
              }
            }
          }
        }
      }
    }
  }

  // Envmap: one importance sample, MIS-weighted against BSDF sampling.
  // Activated for both single-layer (`world_type==1`) and mixed
  // (`world_type==2`) envmaps. Without this gate covering both, mixed
  // worlds would skip envmap NEE entirely — BSDF-sampled paths into the
  // sky would still pick up envmap radiance but with no NEE strategy
  // competing on MIS weights, doubling the effective contribution
  // (previously masquerading as 2× brightness on diffuse surfaces in
  // pabellon's pool basin).
  if (params.world_type != 0 && params.envmap_integral > 0.0f) {
    EnvSample es = sample_envmap(rng);
    if (es.pdf > 0.0f) {
      float3 vis = shadow_transmittance(P, es.dir, 1e20f, vstack);
      if (luminance(vis) > 0.0f) {
        BsdfEval b = eval_bsdf(e, wo, es.dir, cache);
        if (b.pdf > 0.0f) {
          // 3-strategy MIS: balance envmap NEE against the sun cones'
          // delta NEE and the BSDF sample. Without the sun NEE term in
          // the denominator the envmap-CDF samples that landed inside
          // a sun cone (clipped sun pixels in the bake retain a small
          // pdf) double-counted against sun NEE; the asymmetry showed
          // up as a slight over-delivery on lit surfaces near the sun.
          float pdf_sun_at_dir = sum_sun_nee_pdf(es.dir);
          float w = power_heuristic_3(es.pdf, pdf_sun_at_dir, b.pdf);
          L = L + b.f * vis * es.L * (w / es.pdf);
        }
      }
    }
  }

  // Constant-world NEE: cosine-weighted hemisphere sample around the
  // primary closure's shading normal. world_type==0 bypasses the
  // importance-sampled envmap branch above, so without a dedicated NEE
  // path here the world's `world_color × world_strength` only reaches
  // surfaces via BSDF-sampled rays that escape to a miss — observed as
  // 10× under-delivery on a uniform grey world (lone_monk debug,
  // 2026-05-04). Cosine sampling matches Lambertian pdfs so the
  // power-heuristic MIS in `trace_path`'s envmap-miss block balances
  // cleanly with the BSDF side; both strategies sum to a single
  // unbiased estimator.
  if (params.world_type == 0 && e.n > 0) {
    float3 Le = make_f3(params.world_color) * params.world_strength;
    if (luminance(Le) > 0.0f) {
      const MaterialEval &e0 = e.slots[0];
      float u1 = rng.next();
      float u2 = rng.next();
      float r = sqrtf(u1);
      float phi = 2.0f * M_PIf * u2;
      float3 local = make_float3(r * cosf(phi), r * sinf(phi),
                                 sqrtf(fmaxf(0.0f, 1.0f - u1)));
      float3 wi = normalize3(e0.T * local.x + e0.B * local.y + e0.Ns * local.z);
      float NoL = dot3(e0.Ns, wi);
      if (NoL > 0.0f) {
        float pdf_cos = NoL * INV_PIf;
        float3 vis = shadow_transmittance(P, wi, 1e20f, vstack);
        if (luminance(vis) > 0.0f) {
          BsdfEval b = eval_bsdf(e, wo, wi, cache);
          if (b.pdf > 0.0f) {
            float w = power_heuristic(pdf_cos, b.pdf);
            L = L + b.f * vis * Le * (w / pdf_cos);
          }
        }
      }
    }
  }

  return L;
}

// Give the cache one stable call frame, outside trace_path's live state. SSS
// has already finished before this call; no leaf or outgoing direction changes
// until both operations complete. Multiple leaves reuse just the last prepared
// leaf, keeping storage independent of the scene's maximum mixture size.
static __device__ __noinline__ BsdfSample sample_surface_lighting(
    const MaterialEvalGroup &e, float3 P, float3 wo, RNG &rng,
    const VolumeStack &vstack, bool rect_mis, float3 &nee) {
  SurfaceClosureCache cache;
  cache.material = nullptr;
  nee = direct_light(e, P, wo, rng, vstack, rect_mis, cache);
  return sample_bsdf(e, wo, rng, cache);
}

// In-scatter NEE for volume scatter events. Same lights as `direct_light`
// but the BSDF role is played by the volume's phase function and there's no
// `MaterialEval` — the scatter point is in free space inside `vol`.
//
// `wi_world` is the incoming ray direction (so wo_world = -wi_world is the
// "outgoing" direction back toward the scatter event's predecessor); the
// phase function depends on the angle between wi_world and the chosen
// shadow direction. Environment NEE uses MIS against phase sampling, paired
// with the miss-path weight. Rect lights also pair NEE with phase-sampled
// light hits; the other explicit-light and mesh-light blocks do not.
static __device__ __noinline__ float3 direct_light_volume(float3 P, float3 wi_world,
                                              const Volume *vol,
                                              const VolumeStack &vstack,
                                              RNG &rng, bool rect_mis) {
  float3 L = make_float3(0, 0, 0);
  float g = vol->anisotropy;

  auto add = [&](float3 wi, float3 emission, float3 vis) {
    float cos_t = dot3(wi_world, wi);
    float ph = volume_phase_eval(vol, cos_t);
    L = L + vis * emission * ph;
  };

  for (int i = 0; i < params.num_point_lights; i++) {
    PointLight &pl = params.point_lights[i];
    float3 Pl, wi;
    float d;
    float light_weight = sample_point_position(make_f3(pl.position), pl.radius,
                                               pl.sphere, P, rng, Pl, wi, d);
    if (d < 1e-4f || light_weight <= 0)
      continue;
    float3 vis = shadow_transmittance(P, wi, d, vstack);
    if (luminance(vis) <= 0.0f)
      continue;
    add(wi, point_light_emission(pl.shader, make_f3(pl.emission), Pl, -wi, d) * light_weight, vis);
  }
  for (int i = 0; i < params.num_sun_lights; i++) {
    SunLight &sl = params.sun_lights[i];
    float3 dir = make_f3(sl.direction);
    float cos_a = sl.cos_angle;
    float u1 = rng.next();
    float u2 = rng.next();
    float ct = 1.0f - u1 * (1.0f - cos_a);
    float st = sqrtf(fmaxf(0.0f, 1.0f - ct * ct));
    float phi = 2.0f * M_PIf * u2;
    float3 T, B;
    build_frame(dir, T, B);
    float3 wi =
        normalize3(T * (st * cosf(phi)) + B * (st * sinf(phi)) + dir * ct);
    float3 vis = shadow_transmittance(P, wi, 1e20f, vstack);
    if (luminance(vis) <= 0.0f)
      continue;
    add(wi, make_f3(sl.emission), vis);
  }
  for (int i = 0; i < params.num_spot_lights; i++) {
    SpotLight &sp = params.spot_lights[i];
    float3 Pl, wi;
    float d;
    float light_weight = sample_point_position(make_f3(sp.position), sp.radius,
                                               sp.sphere, P, rng, Pl, wi, d);
    if (d < 1e-4f || light_weight <= 0)
      continue;
    float3 center_delta = make_f3(sp.position) - P;
    float cos_a = sp.sphere && dot3(center_delta, center_delta) < sp.radius * sp.radius
                      ? 1.0f : dot3(make_f3(sp.direction), -wi);
    if (cos_a <= sp.cos_outer)
      continue;
    float falloff = 1.0f;
    if (cos_a < sp.cos_inner) {
      float t =
          (cos_a - sp.cos_outer) / fmaxf(sp.cos_inner - sp.cos_outer, 1e-4f);
      falloff = t * t * (3.0f - 2.0f * t);
    }
    float3 vis = shadow_transmittance(P, wi, d, vstack);
    if (luminance(vis) <= 0.0f)
      continue;
    add(wi, point_light_emission(sp.shader, make_f3(sp.emission), Pl, -wi, d) * falloff * light_weight, vis);
  }
  if (params.num_rect_lights > 0) {
    float pmf_light;
    int idx = sample_rect_light(P, make_float3(0, 0, 0), rng.next(), pmf_light);
    if (idx >= 0 && pmf_light > 0.0f) {
      AreaRectLight &ar = params.rect_lights[idx];
      float su = rng.next();
      float sv = rng.next();
      float pdf_solid;
      float3 Pl = sample_rect_position(ar, P, su, sv, pdf_solid);
      float3 ld = Pl - P;
      float d2 = dot3(ld, ld);
      float d = sqrtf(d2);
      if (d > 1e-4f) {
        float3 wi = ld / d;
        float cos_light = -dot3(make_f3(ar.normal), wi);
        if (ar.two_sided != 0u)
          cos_light = fabsf(cos_light);
        if (cos_light > 0.0f) {
          float3 vis = shadow_transmittance(P, wi, d, vstack);
          if (luminance(vis) > 0.0f) {
            float pdf = pdf_solid * pmf_light;
            if (pdf > 0.0f) {
              // See the radiance-path area block: -wi lands on the profile's
              // zero backside because the rect emits along +Z; pass `wi`.
              float ies = ies_lookup(ar.ies_data, ar.ies_n_v, ar.ies_n_h,
                                     ar.light_rotation, wi);
              float phase_pdf = volume_phase_eval(vol, dot3(wi_world, wi));
              float w = ar.camera_visible != 0u && rect_mis ?
                  power_heuristic(pdf, phase_pdf) : 1;
              add(wi, area_light_emission(ar, Pl, -wi, d) * ies * (w / pdf), vis);
            }
          }
        }
      }
    }
  }
  // Mesh lights — same two-stage CDF as the surface NEE block, contribution
  // routed through `add` (phase function in place of BSDF eval). This block
  // does not apply the phase-sampling MIS used by environment NEE below.
  if (params.num_mesh_lights > 0) {
    float u = rng.next();
    int mli = cdf_search(params.mesh_light_cdf, params.num_mesh_lights, u);
    MeshLight &ml = params.mesh_lights[mli];
    float pmf_outer =
        (params.mesh_light_cdf[mli + 1] - params.mesh_light_cdf[mli]);
    if (pmf_outer > 0.0f && ml.num_triangles > 0) {
      float ut = rng.next();
      int ti = cdf_search(ml.cdf, (int)ml.num_triangles, ut);
      MeshLightTri &tri = ml.triangles[ti];
      float pmf_tri = ml.cdf[ti + 1] - ml.cdf[ti];
      if (pmf_tri > 0.0f) {
        float r1 = rng.next();
        float r2 = rng.next();
        float sq = sqrtf(r1);
        float b0 = 1.0f - sq;
        float b1 = sq * (1.0f - r2);
        float b2 = sq * r2;
        float3 v0 = make_f3(tri.v0);
        float3 v1 = make_f3(tri.v1);
        float3 v2 = make_f3(tri.v2);
        float3 Pl = v0 * b0 + v1 * b1 + v2 * b2;
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;
        float3 ng = cross3(e1, e2);
        float ng_len2 = dot3(ng, ng);
        if (ng_len2 > 1e-20f) {
          float3 n = ng * (1.0f / sqrtf(ng_len2));
          float3 ld = Pl - P;
          float d2 = dot3(ld, ld);
          float d = sqrtf(d2);
          if (d > 1e-4f) {
            float3 wi = ld / d;
            float cos_light = -dot3(n, wi);
            if (ml.two_sided != 0u)
              cos_light = fabsf(cos_light);
            if (cos_light > 0.0f) {
              float3 vis = shadow_transmittance(P, wi, d, vstack);
              if (luminance(vis) > 0.0f) {
                float area = tri.area;
                float pdf_area = 1.0f / fmaxf(area, 1e-12f);
                float pdf_solid = pdf_area * d2 / cos_light;
                float pdf = pdf_solid * pmf_outer * pmf_tri;
                if (pdf > 0.0f) {
                  // Per-sample emission radiance — same texture-aware
                  // path as the surface NEE block.
                  float3 Le = mesh_light_emission(tri,b1,b2,-wi,d);
                  add(wi, Le / pdf, vis);
                }
              }
            }
          }
        }
      }
    }
  }
  if (params.world_type != 0 && params.envmap_integral > 0.0f) {
    EnvSample es = sample_envmap(rng);
    if (es.pdf > 0.0f) {
      float3 vis = shadow_transmittance(P, es.dir, 1e20f, vstack);
      if (luminance(vis) > 0.0f) {
        // MIS-weight against phase sampling. After a phase-sampled scatter
        // a missed ray hitting the envmap will be MIS-weighted in the
        // miss-path block (uses prev_bsdf_pdf = phase pdf). To make the
        // estimator unbiased we apply the matching light-side weight here.
        float cos_t = dot3(wi_world, es.dir);
        float ph_pdf = volume_phase_eval(vol, cos_t);
        float w = power_heuristic(es.pdf, ph_pdf);
        // `add` multiplies by phase_eval, which equals ph_pdf, so the
        // contribution is throughput * phase * vis * L * w / pdf_light.
        add(es.dir, es.L * (w / es.pdf), vis);
      }
    }
  }
  return L;
}
