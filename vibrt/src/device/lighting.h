// Cycles packed IES table; position/orientation remain renderer-owned.
static __device__ float ies_lookup(const float *data, unsigned int nv, unsigned int nh,
                                   const float *rotation, float3 direction) {
  if (!data || nv < 2 || nh < 2)
    return 1.0f;
  float3 local = make_float3(
      rotation[0] * direction.x + rotation[3] * direction.y + rotation[6] * direction.z,
      rotation[1] * direction.x + rotation[4] * direction.y + rotation[7] * direction.z,
      rotation[2] * direction.x + rotation[5] * direction.y + rotation[8] * direction.z);
  float length = sqrtf(dot3(local, local));
  if (length == 0)
    return 0;
  float theta = acosf(fminf(1.0f, fmaxf(-1.0f, -local.z / length)));
  float phi = atan2f(local.x, local.y) + M_PIf;
  if (phi >= 2 * M_PIf)
    phi = 0;
  return vibrt_ies::kernel_ies_interp(data, 0, phi, theta);
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

// Spatial selection shares Cycles' bounded importance and branch probabilities.
static __device__ int sample_mesh_light(float3 P, float3 N, float u, float &pmf) {
  return vibrt_light_tree::sample((vibrt_light_tree::KernelGlobals)params.mesh_light_tree,
                                  cycles_vector(P), cycles_vector(N), u, &pmf);
}
static __device__ float mesh_light_pmf(int index, float3 P, float3 N) {
  return vibrt_light_tree::pdf((vibrt_light_tree::KernelGlobals)params.mesh_light_tree,
                               params.mesh_light_leaves[index], cycles_vector(P), cycles_vector(N));
}
static __device__ int sample_rect_light(float3 P, float3 N, float u, float &pmf) {
  return vibrt_light_tree::sample((vibrt_light_tree::KernelGlobals)params.rect_light_tree,
                                  cycles_vector(P), cycles_vector(N), u, &pmf);
}
static __device__ float rect_light_pmf(int index, float3 P, float3 N) {
  return vibrt_light_tree::pdf((vibrt_light_tree::KernelGlobals)params.rect_light_tree,
                               params.rect_light_leaves[index], cycles_vector(P), cycles_vector(N));
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
    float3 vis = shadow_transmittance(P, wi, d, vstack, e.slots[0].Ng);
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
    float3 vis = shadow_transmittance(P, wi, 1e20f, vstack, e.slots[0].Ng);
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
    float3 vis = shadow_transmittance(P, wi, d, vstack, e.slots[0].Ng);
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
          float3 vis = shadow_transmittance(P, wi, d, vstack, e.slots[0].Ng);
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

  // Select a mesh group with the spatial light tree, then a triangle with
  // its power CDF. Cycles chooses solid-angle or area sampling and supplies
  // the matching PDF used by the BSDF-hit MIS path.
  if (params.num_mesh_lights > 0) {
    float pmf_outer;
    int mli = sample_mesh_light(P, e.slots[0].Ns, rng.next(), pmf_outer);
    if (mli >= 0 && pmf_outer > 0) {
      MeshLight &ml = params.mesh_lights[mli];
      if (ml.num_triangles > 0) {
        int ti = cdf_search(ml.cdf, (int)ml.num_triangles, rng.next());
        MeshLightTri &tri = ml.triangles[ti];
        float pmf_tri = ml.cdf[ti + 1] - ml.cdf[ti];
        vibrt_triangle::Triangle geometry = {{cycles_vector(make_f3(tri.v0)),
                                              cycles_vector(make_f3(tri.v1)),
                                              cycles_vector(make_f3(tri.v2))},
                                             ml.two_sided != 0};
        vibrt_triangle::LightSample ls = {};
        if (pmf_tri > 0 && vibrt_triangle::triangle_light_sample<false>(
                               &geometry, 0, 0, vibrt_cycles::make_float2(rng.next(), rng.next()),
                               0, &ls, cycles_vector(P))) {
          float3 wi = make_float3(ls.D.x, ls.D.y, ls.D.z);
          float pdf_light = ls.pdf * pmf_outer * pmf_tri;
          float3 vis = shadow_transmittance(P, wi, ls.t, vstack, e.slots[0].Ng);
          if (pdf_light > 0 && luminance(vis) > 0) {
            auto uv = vibrt_cycles::ray_triangle_uv(cycles_vector(P), ls.D, geometry.vertices[0],
                                                    geometry.vertices[1], geometry.vertices[2]);
            float3 Le = mesh_light_emission(tri, uv.x, uv.y, -wi, ls.t);
            BsdfEval b = eval_bsdf(e, wo, wi, cache);
            L = L + b.f * vis * Le * (power_heuristic(pdf_light, b.pdf) / pdf_light);
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
      float3 vis = shadow_transmittance(P, es.dir, 1e20f, vstack, e.slots[0].Ng);
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
        float3 vis = shadow_transmittance(P, wi, 1e20f, vstack, e.slots[0].Ng);
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
// light hits; mesh-light sampling also pairs with phase-sampled hits.
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
  // Mesh lights use spatial group selection and area sampling inside a volume,
  // with matching phase-sampled-hit MIS in the integrator.
  if (params.num_mesh_lights > 0) {
    float pmf_outer;
    int mli = sample_mesh_light(P, make_float3(0, 0, 0), rng.next(), pmf_outer);
    if (mli >= 0 && pmf_outer > 0) {
      MeshLight &ml = params.mesh_lights[mli];
      if (ml.num_triangles > 0) {
        int ti = cdf_search(ml.cdf, (int)ml.num_triangles, rng.next());
        MeshLightTri &tri = ml.triangles[ti];
        float pmf_tri = ml.cdf[ti + 1] - ml.cdf[ti];
        vibrt_triangle::Triangle geometry = {{cycles_vector(make_f3(tri.v0)),
                                              cycles_vector(make_f3(tri.v1)),
                                              cycles_vector(make_f3(tri.v2))},
                                             ml.two_sided != 0};
        vibrt_triangle::LightSample ls = {};
        if (pmf_tri > 0 &&
            vibrt_triangle::triangle_light_sample<true>(
                &geometry, 0, 0, vibrt_cycles::make_float2(rng.next(), rng.next()), 0, &ls,
                cycles_vector(P)) &&
            (ml.two_sided || vibrt_cycles::dot(ls.Ng, -ls.D) > 0)) {
          float3 wi = make_float3(ls.D.x, ls.D.y, ls.D.z);
          float pdf_light = ls.pdf * pmf_outer * pmf_tri;
          float3 vis = shadow_transmittance(P, wi, ls.t, vstack);
          if (pdf_light > 0 && luminance(vis) > 0) {
            auto uv = vibrt_cycles::ray_triangle_uv(cycles_vector(P), ls.D, geometry.vertices[0],
                                                    geometry.vertices[1], geometry.vertices[2]);
            float3 Le = mesh_light_emission(tri, uv.x, uv.y, -wi, ls.t);
            float p = volume_phase_eval(vol, dot3(wi_world, wi));
            add(wi, Le * (power_heuristic(pdf_light, p) / pdf_light), vis);
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
