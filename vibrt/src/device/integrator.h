static __forceinline__ __device__ float3 clamp_indirect(float3 c,
                                                         float max_sum) {
  if (max_sum <= 0.0f)
    return c;
  // Cycles' `film_clamp_light` compares against `reduce_add(fabs(L))` — the
  // SUM of the spectrum's components — not luminance. For a white
  // contribution this clamps at max_sum/3 (Cycles) vs max_sum (a luminance
  // test), so a luminance test lets ~3× more energy through before clamping;
  // on aggressively-clamped scenes (classroom: clamp_indirect=3) that reads
  // as a broad indirect over-delivery. Match Cycles exactly.
  float s = fabsf(c.x) + fabsf(c.y) + fabsf(c.z);
  return (s > max_sum) ? c * (max_sum / s) : c;
}

// Ray/rect-light intersection: returns closest one-sided front-face hit inside
// [t_min, t_max). Rect-light planes are invisible from behind (Cycles default:
// emission on one side, backside doesn't render). We match NEE's sign
// convention where `ar.normal` points along emission direction.
static __device__ int intersect_rect_lights(float3 origin, float3 dir,
                                            float t_min, float t_max,
                                            float &t_out) {
  int best = -1;
  float t_best = t_max;
  for (int i = 0; i < params.num_rect_lights; i++) {
    AreaRectLight &ar = params.rect_lights[i];
    // Hidden emitters don't occlude camera/specular rays — they only
    // contribute via NEE. Skipping them here also prevents a hidden rect
    // in front of a visible one from shadowing the latter.
    if (ar.camera_visible == 0u)
      continue;
    float3 normal = make_f3(ar.normal);
    float3 corner = make_f3(ar.corner);
    float3 u_axis = make_f3(ar.u_axis);
    float3 v_axis = make_f3(ar.v_axis);
    float denom = dot3(normal, dir);
    // Hit the emissive face when the ray is heading against the normal.
    // Two-sided rects (emissive meshes) emit from both faces, so the
    // sign of `denom` doesn't matter beyond being non-zero.
    if (ar.two_sided == 0u) {
      if (denom >= -1e-6f)
        continue;
    } else {
      if (fabsf(denom) < 1e-6f)
        continue;
    }
    float t = dot3(corner - origin, normal) / denom;
    if (t <= t_min || t >= t_best)
      continue;
    float3 Ph = origin + dir * t;
    float3 d = Ph - corner;
    float su = dot3(d, u_axis);
    float sv = dot3(d, v_axis);
    if (su < 0.0f || su > ar.size_u)
      continue;
    if (sv < 0.0f || sv > ar.size_v)
      continue;
    if (ar.disk) {
      float x = 2 * su / ar.size_u - 1, y = 2 * sv / ar.size_v - 1;
      if (x * x + y * y > 1) continue;
    }
    t_best = t;
    best = i;
  }
  t_out = t_best;
  return best;
}

static __device__ __noinline__ float3 trace_path(float3 origin, float3 dir,
                                                 RNG &rng, float first_ray_tmin,
                                                 float &transparency, float ray_dP, float ray_dD) {
  float3 throughput = make_float3(1, 1, 1);
  float3 L = make_float3(0, 0, 0);
  bool last_specular = true;
  bool camera_path = true;
  transparency = 0;
  float prev_bsdf_pdf = 0.0f;
  // Evaluate the matching NEE PDF at the original scatter point, before
  // ray offsets or transparent volume boundaries move the ray origin.
  float3 prev_nee_position = origin;
  // Last sampled vertex's shading normal: matches rect selection and the
  // constant-world cosine-hemisphere NEE PDF. Zero after volume scattering
  // (no receiver normal and no constant-world hemisphere NEE there).
  float3 prev_normal = make_float3(0.0f, 0.0f, 1.0f);
  // Cycles caps diffuse / glossy / transmission bounces independently of
  // the total path-length cap. Track per-lobe counters; once one runs out
  // the path terminates (diffuse-saturated scenes — lone_monk's brick
  // courtyard with diffuse_bounces=2 — used to receive 4× too much
  // brick-on-brick indirect light because the kernel only honoured the
  // total `max_depth=8`).
  unsigned int diffuse_bounces = 0;
  unsigned int glossy_bounces = 0;
  unsigned int transmission_bounces = 0;
  unsigned int transparent_bounces = 0;
  unsigned int last_path_flags = 1u;
  // Filter-glossy state machine (mirrors Cycles' Light Paths > Filter
  // Glossy). Track the smallest BSDF pdf the path has sampled at a
  // glossy / transmission bounce. The next surface's BSDF α is clamped
  // up by an amount that keeps its lobe at least as wide as
  // 1/(filter_glossy · path_min_pdf) — concretely
  // `min_alpha = sqrt(filter_glossy / path_min_pdf)`. Camera rays start
  // with min_pdf=∞ → min_alpha=0 (no inflation), so primary visibility
  // sees authored materials unmodified; only paths that already passed
  // through a sharp lobe get blurred. Cycles' filter_glossy=0 short-
  // circuits the inflation entirely.
  float path_min_ray_pdf = 1e30f;
  float min_alpha = 0.0f;
  VolumeStack vstack;
  vstack.depth = 0;
  if (params.num_initial_volumes)
    initialize_volume_stack(origin + dir * first_ray_tmin, vstack);

  int camera_skip = 0;
  unsigned int volume_boundaries = 0;
  unsigned int event = 0;
  for (unsigned int bounce = 0; bounce < params.max_depth; bounce++) {
    rng.begin_event(++event);
    PathVertex v;
    v.hit = 0;
    unsigned int hi, lo;
    pack_ptr(&v, hi, lo);
    // Bounce 0 is the camera ray and must honour Cycles' Camera "Clip
    // Start": ignore intersections closer than `first_ray_tmin` from the
    // eye. We do this via OptiX's t_min instead of pushing `origin`
    // forward, because a forward push can land the new origin inside
    // geometry (e.g. flat_archiviz puts the camera ~0.6m from the east
    // wall; pushing 2m forward placed the origin INSIDE that wall and
    // primary rays lost the entire left half of the frame). Subsequent
    // bounces use a small epsilon so reflections / refractions still
    // self-clear.
    //
    // Once we're inside the bounce-0 loop, also advance through back-face
    // hits. Cycles' clip_start is meant to hide a thick wall the camera
    // sits next to (flat_archiviz: 0.35m wall front=1.7m, back=2.05m); a
    // strict t_min skips the front face but we still hit the back face
    // just past it and shade it as a dark interior. The back-face advance
    // mimics Cycles' "skip the whole hidden object" intent.
    float ray_tmin = (bounce == 0 && camera_skip == 0) ? first_ray_tmin : 0.0f;
    // Per-bounce visibility mask: primary radiance (0x01) for camera ray
    // and specular bounce chains, secondary radiance (0x04) for diffuse
    // bounces. Camera-hidden emissive instances drop their 0x01 bit so
    // they appear in the lighting solution (via mesh-light NEE +
    // diffuse-bounce hits) but stay invisible to the camera/specular
    // path — matches Cycles' Ray Visibility → Camera flag semantics.
    unsigned int rad_mask = (bounce == 0 || last_specular) ? 0x01u : 0x04u;
    optixTrace(params.traversable, origin, dir, ray_tmin, 1e20f, 0.0f,
               OptixVisibilityMask(rad_mask), OPTIX_RAY_FLAG_NONE,
               0, // SBT offset (radiance ray type)
               3, // SBT stride (radiance, shadow and query)
               0, // miss index (radiance miss)
               hi, lo);
    if (bounce == 0 && transparent_bounces == 0 && volume_boundaries == 0 &&
        v.hit != 0 && dot3(v.Ng, dir) > 0.0f &&
        (!v.mat || !v.mat->volume)) {
      // Only skip back-faces that are near `clip_start` — the workaround's
      // intent is to hide a wall the camera sits inside (flat_archiviz:
      // 0.35m wall, front at 1.7m=clip_start, back at 2.05m). Distant
      // back-faces (e.g. ies_light's room walls authored with normals
      // facing outward, hit at ~5.9m through clear air) are NOT what we
      // want to skip — doing so makes the camera ray miss the entire
      // scene and renders pitch black. Use a 1m thickness budget past
      // `clip_start`: enough for a thick interior wall, tight enough to
      // not eat any genuine back-facing geometry past the camera region.
      float t_hit = sqrtf(dot3(v.P - origin, v.P - origin));
      bool near_clip = t_hit < first_ray_tmin + 1.0f;
      if (near_clip && camera_skip < 4) {
        // Back-face hit on the camera ray: step past it and retry without
        // counting this as a bounce. Cap the retry depth so a pathological
        // all-back-faces direction can't loop forever.
        origin = offset_surface(v.P, v.Ng, dir);
        ray_dP += v.ray_length * ray_dD;
        camera_skip++;
        bounce--;  // wraps to UINT_MAX, the for-loop's bounce++ brings it back to 0
        continue;
      }
      // Cap exhausted (or back-face is far from clip_start so we treat it
      // as a regular interior hit) — flag it so the host can warn after the
      // launch, then fall through and shade this back-face. One increment
      // per sample-per-pixel where the cap is the reason we fall through;
      // distant back-faces are silent because the eval_material path below
      // flips Ns and shades them like a normal interior surface.
      if (near_clip && params.primary_back_face_skip_exhausted != nullptr) {
        atomicAdd(params.primary_back_face_skip_exhausted, 1u);
      }
    }

    // Geometry hit distance (∞ on miss). Needed so a rect light only counts
    // when it occludes geometry rather than being behind a wall.
    float t_geom = 1e20f;
    if (v.hit != 0) {
      float3 delta = v.P - origin;
      t_geom = sqrtf(dot3(delta, delta));
    }
    float t_rect;
    int rect_idx = intersect_rect_lights(origin, dir, ray_tmin, t_geom, t_rect);
    float t_segment = (rect_idx >= 0) ? t_rect : t_geom;

    // A bounded medium cannot extend to infinity. Roundoff at an edge or
    // an open boundary can miss the exit triangle; don't integrate a finite
    // object's emission over the miss sentinel distance.
    if (v.hit == 0 && rect_idx < 0 && vstack.depth > 0)
      vstack.depth = 0;

    // Volume distance sampling: if we're currently inside a volume, decide
    // whether the ray scatters before reaching the next surface/light. Path
    // emission is integrated in closed form over the entire segment so
    // emissive media don't add MC noise.
    const Volume *cur_vol = volume_stack_top(vstack);
    if (cur_vol != nullptr) {
      float volume_start = (bounce == 0 && camera_skip == 0) ? first_ray_tmin : 0.0f;
      float volume_length = fmaxf(t_segment - volume_start, 0.0f);
      float sigma_t_avg = vol_sigma_t_avg(cur_vol);
      // Emission is integrated over the full segment independently of the
      // random scattering distance, including the endpoint survival branch.
      L = L + throughput * vol_path_emission(cur_vol, volume_length, last_path_flags);
      if (sigma_t_avg > 1e-12f) {
        vibrt_volume::Ray volume_ray{cycles_vector(origin + dir * volume_start), cycles_vector(dir)};
        vibrt_volume::EquiangularCoefficients coeffs;
        bool equiangular = luminance(make_f3(cur_vol->sigma_s)) > 0 &&
            volume_equiangular_coefficients(origin + dir * volume_start, dir, volume_length, rng, coeffs);
        float u = rng.next();
        float t_scatter, pdf_angular = 0;
        if (equiangular && u < 0.5f)
          t_scatter = vibrt_volume::volume_equiangular_sample(&volume_ray, coeffs, u * 2, &pdf_angular);
        else {
          t_scatter = -logf(fmaxf(1 - (equiangular ? (u - 0.5f) * 2 : u), 1e-30f)) / sigma_t_avg;
          if (equiangular && t_scatter < volume_length)
            pdf_angular = vibrt_volume::volume_equiangular_pdf(&volume_ray, coeffs, t_scatter);
        }
        if (t_scatter < volume_length) {
          // --- Volume scatter event ---
          float3 P_scatter = origin + dir * (volume_start + t_scatter);
          float3 tr = vol_beam_transmittance(cur_vol, t_scatter);
          float pdf_t = sigma_t_avg * expf(-sigma_t_avg * t_scatter);
          if (equiangular)
            pdf_t = 0.5f * (pdf_t + pdf_angular);
          float3 sigma_s = make_float3(cur_vol->sigma_s[0],
                                       cur_vol->sigma_s[1],
                                       cur_vol->sigma_s[2]);
          throughput = throughput * tr * sigma_s / fmaxf(pdf_t, 1e-30f);

          // In-scatter NEE: phase-weighted contribution from each light.
          float3 nee = throughput * direct_light_volume(P_scatter, dir,
              cur_vol, vstack, rng, bounce + 1 < params.max_depth);
          if (bounce > 0)
            nee = clamp_indirect(nee, params.clamp_indirect);
          L = L + nee;

          // Sample a direction from the volume's selected phase function.
          // Build the frame around the previous direction so wi_local.z
          // corresponds to forward scatter.
          float u1 = rng.next();
          float u2 = rng.next();
          float pdf_phase;
          float3 wi_local = volume_phase_sample(cur_vol, u1, u2, pdf_phase);
          float3 T_frame, B_frame;
          build_frame(dir, T_frame, B_frame);
          float3 wi = T_frame * wi_local.x + B_frame * wi_local.y +
                      dir * wi_local.z;
          // Phase sampling is exact (sampled from p / pdf == 1), so no
          // additional throughput multiplier is needed.
          last_specular = false;
          prev_bsdf_pdf = pdf_phase;
          prev_nee_position = P_scatter;
          prev_normal = make_float3(0, 0, 0);

          // Start roulette at the first scattering event: fewer long
          // paths, at the cost of higher variance at a fixed sample count.
          { // Roulette also applies after the first scattering event.
            float q = fminf(fmaxf(luminance(throughput), 0.05f), 0.95f);
            if (rng.next() > q)
              break;
            throughput = throughput / q;
          }
          origin = P_scatter;
          ray_dP += (volume_start + t_scatter) * ray_dD;
          dir = wi;
          continue;
        }
        // No scatter — path emission across the full segment, then apply
        // beam transmittance to the endpoint and divide by survival prob so
        // the no-scatter branch stays an unbiased estimator.
        float3 tr = vol_beam_transmittance(cur_vol, volume_length);
        float pdf_surv = expf(-sigma_t_avg * volume_length);
        if (equiangular) pdf_surv *= 0.5f;
        throughput = throughput * tr / fmaxf(pdf_surv, 1e-30f);
      }
    }

    if (rect_idx >= 0) {
      // Hit a camera-visible area light in front of any geometry. Hidden
      // rects are pre-filtered by intersect_rect_lights and never reach
      // here, so they don't occlude primary/specular rays. Non-delta hits
      // use the PDF of the matching NEE proposal at the previous vertex.
      AreaRectLight &ar = params.rect_lights[rect_idx];
      // IES applies to direct camera-ray hits too — Cycles' IES Texture
      // multiplies emission for every visibility class, not just NEE.
      // `dir` is camera→light here; the IES table is sampled in the
      // light's local frame.
      float ies = ies_lookup(ar.ies_data, ar.ies_n_v, ar.ies_n_h,
                             ar.light_rotation, dir);
      float3 em = throughput * area_light_emission(ar, origin + dir * t_rect, -dir, t_rect, last_path_flags) * ies;
      if (bounce == 0 || last_specular) {
        if (bounce > 0)
          em = clamp_indirect(em, params.clamp_indirect);
        L = L + em;
      } else if (prev_bsdf_pdf > 0.0f) {
        // BSDF sampled this (camera-visible) area light on a non-specular
        // bounce. MIS-weight against the NEE sampler (area-light solid-angle
        // pdf × rect-select pmf) — the piece NEE can't capture for a sharp
        // glossy lobe reflecting an area light.
        float pmf = rect_light_pmf(rect_idx, prev_nee_position, prev_normal);
        float pdf_light = rect_direction_pdf(ar, prev_nee_position,
                                              origin + dir * t_rect) * pmf;
        float w_bsdf = power_heuristic(prev_bsdf_pdf, pdf_light);
        em = clamp_indirect(em * w_bsdf, params.clamp_indirect);
        L = L + em;
      }
      break;
    }

    if (v.hit == 0) {
      if (params.film_transparent && camera_path) {
        transparency += (throughput.x + throughput.y + throughput.z) / 3;
        break;
      }
      // is_camera_ray for the world-bg split: bounce==0 is the primary
      // ray, and a chain of specular bounces (mirror / glass) preserves
      // the camera-ray semantic in Cycles' Light Path classification.
      bool is_camera_ray = (bounce == 0) || last_specular;
      float3 bg = world_background(dir, is_camera_ray);
      float w = 1.0f;
      if (bounce > 0 && !last_specular) {
        if (params.world_type != 0) {
          // 3-strategy MIS: BSDF sample vs envmap NEE vs sun NEE. The
          // sun term matters when the BSDF-sampled ray happens to land
          // inside a sun cone — the clipped envmap pixels there have
          // small `envmap_pdf` so the original 2-strategy MIS gave the
          // BSDF side full weight and added the (clipped, ~13×) sun
          // pixel value, double-counting on top of the dedicated sun
          // MIS branch a few lines below.
          float p_env = envmap_pdf(dir);
          float p_sun = sum_sun_nee_pdf(dir);
          w = power_heuristic_3(prev_bsdf_pdf, p_env, p_sun);
        } else {
          // Constant world: NEE samples cosine-weighted hemisphere with
          // pdf = max(0, NoL) / π. MIS against the BSDF sample's pdf
          // so the two strategies (NEE in `direct_light` and BSDF-miss
          // here) sum to a single unbiased estimator. Without this
          // weight the constant-world NEE addition double-counts the
          // sky on every diffuse miss path.
          float NoL = dot3(prev_normal, dir);
          float pdf_cos = (NoL > 0.0f) ? NoL * INV_PIf : 0.0f;
          w = power_heuristic(prev_bsdf_pdf, pdf_cos);
        }
      }
      float3 bg_contrib = throughput * bg * w;
      if (bounce > 0)
        bg_contrib = clamp_indirect(bg_contrib, params.clamp_indirect);
      L = L + bg_contrib;

      // Sun-light MIS BSDF-side branch: when a BSDF-sampled scatter ray
      // happens to land inside a SunLight's cone, deliver the sun's
      // emission weighted against the cone-sampler pdf
      // (`pdf_sun = 1/sun_omega`). Sun is removed from the envmap during
      // extraction, so `bg` above is the residual sky only — without
      // this branch BSDF samples that hit the sun direction would carry
      // ~zero radiance and miss the highlight that NEE alone leaves
      // noisy on glossy/specular surfaces aimed at the sun. Skipped on
      // camera/specular rays where w_bsdf = 1 already (NEE didn't fire
      // on delta lobes at the previous vertex, so the sun's emission
      // would arrive unweighted from the surface-hit branch instead —
      // but the sun isn't a surface we hit in BLAS, so even there the
      // contribution would be lost without this code).
      for (int i = 0; i < params.num_sun_lights; i++) {
        SunLight &sl = params.sun_lights[i];
        float3 sun_dir = make_f3(sl.direction);
        float cos_to_sun = dot3(dir, sun_dir);
        if (cos_to_sun <= sl.cos_angle)
          continue;
        // Inside this sun's cone — deliver Le_per_dir × MIS weight.
        // The kernel's `emission` field stores `Le × sun_omega`, so
        // we recover Le by dividing back out.
        float sun_omega = 2.0f * M_PIf * fmaxf(1.0f - sl.cos_angle, 1e-30f);
        float pdf_sun = 1.0f / sun_omega;
        float w_bsdf;
        if (bounce == 0 || last_specular) {
          // Camera ray or after specular chain: NEE didn't fire on the
          // delta lobe at the previous vertex (or there was no previous
          // vertex), so the sun's emission arrives unweighted here.
          w_bsdf = 1.0f;
        } else {
          // 3-strategy MIS: BSDF sample vs sun NEE vs envmap NEE. The
          // env term is small (clipped sun pixels) but non-zero; the
          // existing 2-strategy form would be biased when the BSDF
          // happens to sample the sun direction and the envmap CDF
          // could have plausibly delivered the same direction.
          float p_env = sum_env_nee_pdf(dir);
          w_bsdf = power_heuristic_3(prev_bsdf_pdf, pdf_sun, p_env);
        }
        float3 Le_sun = make_f3(sl.emission) / sun_omega;
        float3 sun_contrib = throughput * Le_sun * w_bsdf;
        if (bounce > 0)
          sun_contrib = clamp_indirect(sun_contrib, params.clamp_indirect);
        L = L + sun_contrib;
      }
      break;
    }

    // Volume boundaries do not consume surface bounces or change ray flags.
    if (v.mat != nullptr && v.mat->volume != nullptr &&
        v.mat->volume_only != 0) {
      bool entering = dot3(v.Ng, dir) < 0.0f;
      if (entering)
        volume_stack_push(vstack, v.mat->volume);
      else
        volume_stack_pop(vstack, v.mat->volume);
      origin = offset_surface(v.P, v.Ng, dir);
      ray_dP += v.ray_length * ray_dD;
      if (++volume_boundaries > 128) break;
      if (bounce == 0) camera_skip++;
      bounce--; // the loop increment restores the same surface bounce
      continue;
    }

#if VIBRT_SHADER_OPS
    // Regroup divergent paths by material before interpreting node graphs.
    // Hint collisions only reduce coherence; they cannot change shading.
    // OptiX treats this as a no-op on devices without reordering support.
    unsigned long long material_key = (unsigned long long)v.mat;
    optixReorder((unsigned int)((material_key >> 8) ^ (material_key >> 16)), 8);
#endif

    // Stack-allocated leaves of the material's binary Mix tree. Lifetime
    // covers every direct_light / sample_bsdf call that reads `e_group`.
    MaterialEval e_slots[MAX_LEAVES];
    v.shader_seed = vibrt_cycles::hash_uint2(rng.state, rng.sample);
    v.path_depth = bounce;
    v.diffuse_depth = diffuse_bounces;
    v.glossy_depth = glossy_bounces;
    v.transmission_depth = transmission_bounces;
    v.path_flags = last_path_flags;
    ray_dP = vibrt_differential::differential_transfer_compact(ray_dP, cycles_vector(dir), ray_dD, v.ray_length);
    surface_differentials(&v, ray_dP);
    float e_weights[MAX_LEAVES];
    // View cosine for view-dependent (Fresnel) mix factors. `dir` is the
    // incoming ray (origin→hit); |dot(Ns, dir)| is the incidence cosine.
    float NoV_mix = fabsf(dot3(v.Ns, dir));
    int e_n = eval_material(v, e_slots, e_weights, MAX_LEAVES, min_alpha, NoV_mix);
    MaterialEvalGroup e_group = {e_slots, e_weights, e_n};
#if VIBRT_AOVS
    if (camera_path && v.mat->num_aov_shaders) {
      float transparent = 0;
      for (int i = 0; i < e_n; i++)
        transparent += e_weights[i] * (1 - e_slots[i].alpha_blend);
      write_shader_aovs(v, 1 - fminf(transparent, 1.0f));
    }
#endif
#if (VIBRT_CLOSURE_TYPES & (1u << 3))
    if (params.film_transparent && camera_path) {
      float holdout = 0;
      for (int i = 0; i < e_n; i++)
        if (e_slots[i].mat->closure_type == 3)
          holdout += e_weights[i];
      transparency += (throughput.x + throughput.y + throughput.z) *
                      (fminf(holdout, 1.0f) / 3);
    }
#endif
    // Flip normal if ray hit backside (for dielectric transmission). Mirror
    // the bitangent instead of rebuilding the frame so the authored tangent
    // rotation (anisotropy axis) survives on back-faces; negating one axis
    // keeps (T, B, Ns) right-handed. Also invert ior: sample_bsdf and
    // eval_bsdf read it as η_t/η_i for the current incident side, so on a
    // back-face hit (ray exiting the medium) we need 1/ior to get correct
    // Fresnel and TIR. Apply uniformly across every leaf — they all share
    // the surface; only per-lobe params differ.
    if (dot3(v.Ns, -dir) < 0.0f) {
      for (int i = 0; i < e_n; ++i) {
        e_slots[i].Ns = -e_slots[i].Ns;
        e_slots[i].B = -e_slots[i].B;
        e_slots[i].ior = 1.0f / e_slots[i].ior;
        e_slots[i].coat_normal = -e_slots[i].coat_normal;
      }
    }
    // Build a scalar/aggregate `e` that legacy code paths can read for
    // surface-level properties (Ns / T / B / ior / mat / emission). The
    // per-leaf eval/sample paths consume `e_group` directly.
    MaterialEval e = e_slots[0];
    {
      float3 em = make_float3(0.0f, 0.0f, 0.0f);
      for (int i = 0; i < e_n; ++i)
        em = em + e_slots[i].emission * e_weights[i];
      e.emission = em;
    }

    // Emission gate. Three cases:
    //   1. Camera ray (bounce == 0) or after a specular chain
    //      (last_specular): NEE didn't sample this surface (no NEE on
    //      delta lobes), so the BSDF is the only path and we add full
    //      emission unweighted.
    //   2. Diffuse bounce that landed on a registered mesh-light triangle:
    //      NEE *did* sample this lamp, so we MIS-weight the BSDF-side
    //      contribution against the NEE-side. Power heuristic with β=2,
    //      same as the envmap miss block. Without this branch the BSDF
    //      side was wasted (legacy gate suppressed it) and noisy small
    //      lamps were sampled at NEE's mercy alone.
    //   3. Diffuse bounce on an emissive surface that *isn't* in any
    //      mesh-light (textured emission, camera-hidden rect-proxy
    //      meshes, or any constant-emission material on an unregistered
    //      object): emission is dropped — the same legacy behaviour
    //      that's been in place since before mesh-light NEE existed.
    //      Promoting more meshes through the mesh-light path is the
    //      proper fix; double-counting them here without NEE coverage
    //      would over-deliver.
    if (bounce == 0 || last_specular) {
      float3 e_contrib = throughput * e.emission;
      if (bounce > 0)
        e_contrib = clamp_indirect(e_contrib, params.clamp_indirect);
      L = L + e_contrib;
    } else if (v.mesh_light_idx >= 0 && v.mat != nullptr
               && v.light_pdf > 0.0f
               && luminance(e.emission) > 0.0f
               && params.mesh_light_total_power > 0.0f
               && v.prim_area > 1e-12f
               && prev_bsdf_pdf > 0.0f) {
      float3 ld = v.P - prev_nee_position;
      float distance = sqrtf(dot3(ld, ld));
      const HitGroupData *hg = v.geometry;
      vibrt_triangle::Triangle triangle;
      triangle.two_sided = params.mesh_lights[v.mesh_light_idx].two_sided != 0;
      for (int k = 0; k < 3; ++k) {
        int index = hg->indices[3 * v.primitive + k];
        triangle.vertices[k] =
            cycles_vector(transform_point(hg->object_to_world, make_f3(hg->vertices + 3 * index)));
      }
      vibrt_triangle::ShaderData sd;
      sd.P = cycles_vector(v.P);
      sd.Ng = cycles_vector(v.Ng);
      sd.wi = cycles_vector(-ld / distance);
      float pdf_solid =
          dot3(prev_normal, prev_normal) == 0
              ? vibrt_triangle::triangle_light_pdf_area_sampling(sd.Ng, sd.wi, distance) /
                    v.prim_area
              : vibrt_triangle::triangle_light_pdf(&triangle, &sd, distance);
      float pmf = v.prim_area * v.light_pdf *
                  mesh_light_pmf(v.mesh_light_idx, prev_nee_position, prev_normal);
      float pdf_light = pdf_solid * pmf;
      float w_bsdf = power_heuristic(prev_bsdf_pdf, pdf_light);
      float3 e_contrib = clamp_indirect(throughput * e.emission * w_bsdf, params.clamp_indirect);
      L = L + e_contrib;
    }

    // Select a BSSRDF independently of local reflection/transmission. Emission
    // above belongs to the entry surface and must not move to the exit point.
#if (VIBRT_CLOSURE_TYPES & (1u << 7))
    bool has_sss = false;
    for (int i=0;i<e_n;i++)
      has_sss |= e_slots[i].mat->sss_method != 0 || e_slots[i].mat->closure_type == 7;
    if (has_sss) {
      float total = 0;
      for (int i=0;i<e_n;i++) total += e_weights[i];
      if (!(total > 0)) break;
      float pick = rng.next()*total;
      int chosen = e_n-1;
      for (int i=0;i<e_n;i++) {
        pick -= e_weights[i];
        if (pick <= 0) {chosen=i;break;}
      }
      e_slots[0]=e_slots[chosen];e_weights[0]=1;e_n=1;e_group.n=1;
      throughput=throughput*total;
      MaterialEval &sub=e_slots[0];
      bool standalone = sub.mat->closure_type == 7;
      float strength = 1, other = 0;
      float3 sss_weight = make_float3(1, 1, 1);
      if (!standalone) {
        vibrt_surface::Closures closures;
        auto p = surface_parameters(sub, -dir);
        vibrt_surface::setup(p, closures);
        strength = vibrt_cycles::average(closures.subsurface_weight) * sub.alpha_blend;
        other = closures.total * sub.alpha_blend + (1 - sub.alpha_blend);
        // The BSSRDF walk/profile already applies its base color.
        auto weight = vibrt_cycles::safe_divide_color(closures.subsurface_weight, p.color);
        sss_weight = make_float3(weight.x, weight.y, weight.z) * sub.alpha_blend;
      }
      float probability=strength/fmaxf(strength+other,1e-12f);
      if (strength>0 && rng.next()<probability) {
        throughput=throughput*(sss_weight/probability);
        bool burley=sub.mat->sss_method<=1;
        float radius=fmaxf(fmaxf(sub.sss_radius.x,sub.sss_radius.y),sub.sss_radius.z)*sub.sss_scale;
        if (radius>1e-6f && !(burley && diffuse_bounces>0)) {
          bool success=burley ? scatter_subsurface(v,sub,rng,throughput) :
                                scatter_subsurface_walk(v,sub,rng,throughput);
          if (!success) break;
        }
        sub.sss_exit=true;
        dir=-sub.Ns;
        camera_path=false;
      } else {
        if (probability>=1 || standalone) break;
        throughput=throughput/(1-probability);
      }
      e=e_slots[0];
    }
#endif
    // NEE
    float3 wo = -dir;
    float3 nee;
    BsdfSample bs = sample_surface_lighting(e_group, v.P, wo, rng, vstack,
                                            bounce + 1 < params.max_depth, nee);
    nee = throughput * nee;
    // bounce==0 → camera-ray NEE, clamp via Cycles' sample_clamp_direct.
    // bounce>0 → indirect-bounce NEE, clamp via sample_clamp_indirect.
    // Cycles applies clamping at exactly the same path-vertex layering.
    if (bounce == 0)
      nee = clamp_indirect(nee, params.clamp_direct);
    else
      nee = clamp_indirect(nee, params.clamp_indirect);
    L = L + nee;

    // Use the BSDF sample for the next bounce.
    if (bs.pdf <= 0.0f)
      break;
    float3 contrib = bs.f / bs.pdf;
    throughput = throughput * contrib;
    camera_path = camera_path && bs.transparent;
    if (bs.transparent) {
      // Transparent BSDF is a null event: preserve camera/path flags, MIS
      // state and surface bounce budgets, just as volume boundaries do.
      if (++transparent_bounces > params.max_transparent_bounces) break;
      if (v.mat->volume) {
        if (dot3(v.Ng, dir) < 0) volume_stack_push(vstack, v.mat->volume);
        else volume_stack_pop(vstack, v.mat->volume);
      }
      origin = offset_surface(v.P, v.Ng, bs.wi);
      dir = bs.wi;
      if (bounce == 0) camera_skip++;
      bounce--;
      continue;
    }
#if (VIBRT_CLOSURE_TYPES & (1u << 4))
    if (bs.portal) {
      origin = bs.portal_position + bs.wi * 1e-4f;
      dir = bs.wi;
      last_specular = true;
      last_path_flags = 1u | 16u | 64u;
      continue;
    }
#endif
    last_specular = bs.specular;
    last_path_flags=(bs.lobe==LOBE_DIFFUSE?4u:bs.lobe==LOBE_GLOSSY?8u:0u)|(bs.specular?16u:0u)|(dot3(bs.wi,v.Ng)*dot3(-dir,v.Ng)>0?32u:64u);
    prev_bsdf_pdf = bs.pdf;
    prev_nee_position = v.P;
    prev_normal = e_slots[0].Ns;

    // Cycles-style per-lobe bounce caps. Cycles' `diffuse_bounces=N`
    // means up to N diffuse reflections may appear in the path, which
    // makes NEE fire at the surface that the Nth bounce LANDS on.
    // Translating that to our sample-then-trace loop: we allow the
    // counter to reach N (NEE at the next iteration is the Nth-bounce
    // indirect contribution), and only terminate when the *next* bounce
    // of the same lobe would exceed N — i.e. on count > limit. Using
    // `>=` here (the obvious-looking choice) under-samples by one bounce.
    if (bs.lobe == LOBE_DIFFUSE) {
      diffuse_bounces++;
      if (diffuse_bounces > params.max_diffuse_bounces)
        break;
    } else if (bs.lobe == LOBE_GLOSSY) {
      glossy_bounces++;
      if (glossy_bounces > params.max_glossy_bounces)
        break;
    } else if (bs.lobe == LOBE_TRANSMISSION) {
      transmission_bounces++;
      if (transmission_bounces > params.max_transmission_bounces)
        break;
    }
    // Update filter-glossy state (Cycles-exact). Cycles tracks the
    // minimum bsdf pdf along the path across *all* non-delta bounces —
    // diffuse included, because the broad diffuse pdf (~NoL/π) is exactly
    // what drops `min_ray_pdf` and triggers blurring of glossy lobes seen
    // downstream (the diffuse→glossy→small-light caustic paths that
    // otherwise firefly). `filter_glossy` holds the UI `blur_glossy`
    // value: blur_pdf = min_ray_pdf / blur_glossy, and the roughness (α)
    // floor is 0.5·sqrt(1 − blur_pdf) when blur_pdf < 1 (matches Cycles'
    // `surface_shader_bsdf_blur`). Near-mirror / smooth-glass bounces
    // carry a huge pdf so `fminf` leaves min_ray_pdf untouched; camera
    // rays start at min_ray_pdf = 1e30 → no blur on primary visibility.
    if (params.filter_glossy > 0.0f) {
      path_min_ray_pdf = fminf(path_min_ray_pdf, bs.pdf);
      float blur_pdf = path_min_ray_pdf / params.filter_glossy;
      if (blur_pdf < 1.0f) {
        float blur_roughness = 0.5f * sqrtf(fmaxf(0.0f, 1.0f - blur_pdf));
        min_alpha = fmaxf(min_alpha, blur_roughness);
      }
    }

    // Start roulette at the first surface event. Survival compensation
    // preserves expected radiance while trading more noise for less tracing.
    { // Roulette also applies after the first scattering event.
      float q = fminf(fmaxf(luminance(throughput), 0.05f), 0.95f);
      if (rng.next() > q)
        break;
      throughput = throughput / q;
    }

    // If the surface has a volume container (Surface + Volume both
    // authored — e.g. coloured glass with subsurface fog), update the
    // volume stack on transmission. Reflection keeps us on the same side.
    if (v.mat != nullptr && v.mat->volume != nullptr) {
      bool was_outside_to_inside = dot3(v.Ng, dir) < 0.0f;
      bool now_outside_to_inside = dot3(v.Ng, bs.wi) < 0.0f;
      if (was_outside_to_inside != now_outside_to_inside) {
        if (now_outside_to_inside)
          volume_stack_push(vstack, v.mat->volume);
        else
          volume_stack_pop(vstack, v.mat->volume);
      }
    }

    origin = offset_surface(v.P, v.Ng, bs.wi);
    dir = bs.wi;
  }
  return L;
}

extern "C" __global__ void __raygen__rg() {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  unsigned int pixel = idx.y * dim.x + idx.x;

  float *adaptive = params.adaptive_buffer ? params.adaptive_buffer + 9 * pixel : nullptr;
  if (params.adaptive_phase) {
    if (params.adaptive_phase == 1 && __float_as_uint(adaptive[8]) >= params.adaptive_min_samples)
      vibrt_adaptive::film_adaptive_sampling_convergence_check(0, params.adaptive_buffer, idx.x, idx.y,
                                                               params.adaptive_threshold, 0, 0, dim.x);
    else if (params.adaptive_phase == 2 && idx.x == 0)
      vibrt_adaptive::film_adaptive_sampling_filter_x(0, params.adaptive_buffer, idx.y, 0, dim.x, 0, dim.x);
    else if (params.adaptive_phase == 3 && idx.y == 0)
      vibrt_adaptive::film_adaptive_sampling_filter_y(0, params.adaptive_buffer, idx.x, 0, dim.y, 0, dim.x);
    return;
  }
  if (adaptive && adaptive[7] != 0)
    return;

  float3 eye = make_f3(params.cam_eye);
  float3 U = make_f3(params.cam_u);
  float3 V = make_f3(params.cam_v);
  float3 W = make_f3(params.cam_w);

  // Denoiser guide AOVs + Mist/Z support: one un-jittered primary ray into
  // the first surface. Keeps guides crisp (no sub-pixel averaging) and
  // cheap (one trace/pixel). The depth_aov drives the addon's Mist pass so
  // Cycles-authored compositors (mist haze, depth-based masks) work on top
  // of vibrt's output; misses store `cam_clip_end` so Mist saturates to 1.
  if (params.sample_start == 0 &&
      (params.albedo_aov != nullptr || params.normal_aov != nullptr ||
       params.depth_aov != nullptr)) {
    float px0 = (2.0f * ((float)idx.x + 0.5f) / (float)dim.x) - 1.0f
                + 2.0f * params.cam_shift_x;
    float py0 = (2.0f * ((float)idx.y + 0.5f) / (float)dim.y) - 1.0f
                + 2.0f * params.cam_shift_y;
    float3 dir0 = params.cam_orthographic ? W : normalize3(U * px0 + V * py0 + W);
    float3 camera_origin = params.cam_orthographic ? eye + U * px0 + V * py0 : eye;
    PathVertex v;
    v.hit = 0;
    // Skip volume-only boundaries on the AOV ray — the denoiser guides
    // should describe the *visible* surface (the wall behind a smoke
    // cube), not the invisible volume container. Capped to a few
    // iterations so a pathological scene can't loop forever. The first
    // iteration honours `cam_clip_start` via t_min so the AOV agrees
    // with what the SPP loop below sees; later iterations use a small
    // epsilon to clear the previous boundary surface.
    float3 origin = camera_origin;
    float ray_tmin = params.cam_clip_start;
    for (int i = 0; i < VOL_STACK_MAX + 2; i++) {
      v.hit = 0;
      unsigned int hi, lo;
      pack_ptr(&v, hi, lo);
      optixTrace(params.traversable, origin, dir0, ray_tmin, 1e20f, 0.0f,
                 OptixVisibilityMask(0x01), OPTIX_RAY_FLAG_NONE, 0, 3, 0, hi, lo);
      if (v.hit == 0)
        break;
      if (v.mat == nullptr || v.mat->volume == nullptr ||
          v.mat->volume_only == 0)
        break;
      origin = offset_surface(v.P, v.Ng, dir0);
      ray_tmin = 0.0f;
    }
    float3 alb = make_float3(0, 0, 0);
    float3 nrm = make_float3(0, 0, 0);
    // Depth alone only needs the intersection; evaluating material nodes and
    // bump derivatives here is useful only when denoiser guides are requested.
    if (v.hit != 0 && (params.albedo_aov != nullptr || params.normal_aov != nullptr)) {
      // AOV pass only reads albedo / normal — the secondary's color is
      // not blended in here (would smear the denoiser's albedo guide on
      // multi-lobe materials). Ask eval_material for just the leftmost
      // leaf (max_leaves=1) so the AOV reads the primary closure's
      // colour verbatim — same Cycles approximation Cycles uses for its
      // denoiser albedo guide.
      MaterialEval e_aov_slots[1];
      float e_aov_weights[1];
      int n = eval_material(v, e_aov_slots, e_aov_weights, 1);
      ShaderEvalCache cache;cache.nodes=0;
      MaterialEval e = (n > 0) ? e_aov_slots[0] : eval_material_one(v, v.mat, cache);
      if (dot3(e.Ns, -dir0) < 0.0f)
        e.Ns = -e.Ns;
      alb = e.base_color;
      float3 Uu = normalize3(U);
      float3 Vu = normalize3(V);
      // W is already unit (camera forward).
      nrm = make_float3(dot3(e.Ns, Uu), dot3(e.Ns, Vu), dot3(e.Ns, W));
    }
    if (params.albedo_aov != nullptr) {
      float *a = &params.albedo_aov[pixel * 3];
      a[0] = alb.x;
      a[1] = alb.y;
      a[2] = alb.z;
    }
    if (params.normal_aov != nullptr) {
      float *n = &params.normal_aov[pixel * 3];
      n[0] = nrm.x;
      n[1] = nrm.y;
      n[2] = nrm.z;
    }
    if (params.depth_aov != nullptr) {
      float d;
      if (v.hit != 0) {
        float3 d3 = v.P - camera_origin;
        d = sqrtf(dot3(d3, d3));
      } else {
        d = params.cam_clip_end;
      }
      params.depth_aov[pixel] = d;
    }
  }

  float3 accum = make_float3(0, 0, 0);
  float transparency_sum = 0;
  if (params.sample_start != 0 && params.accumulation != nullptr) {
    accum = make_f3(&params.accumulation[pixel * 4]);
    transparency_sum = params.accumulation[pixel * 4 + 3];
  }
  const unsigned int sample_start = adaptive ? __float_as_uint(adaptive[8]) : params.sample_start;
  const unsigned int sample_end = sample_start + params.sample_count;
  for (unsigned int s = sample_start; s < sample_end; s++) {
    RNG rng(pixel, s, 0u);
    float jx = vibrt_camera::lookup_table_read(params.cam_filter_table, rng.next(), 0, 1024);
    float jy = vibrt_camera::lookup_table_read(params.cam_filter_table, rng.next(), 0, 1024);
    float px = (2.0f * ((float)idx.x + jx) / (float)dim.x) - 1.0f
               + 2.0f * params.cam_shift_x;
    float py = (2.0f * ((float)idx.y + jy) / (float)dim.y) - 1.0f
               + 2.0f * params.cam_shift_y;
    float3 dir = params.cam_orthographic ? W : normalize3(U * px + V * py + W);
    // Honour Cycles' Camera "Clip Start" via t_min on the first ray
    // inside trace_path. We deliberately do NOT push `origin` forward:
    // when the camera sits close to a wall on one side (flat_archiviz
    // ~0.6m east wall, lens=50mm, clip_start=2m), shifting the origin
    // by `dir * clip_start` lands inside the wall for rays going that
    // way, and OptiX returns no hit → a hard black corner on that side
    // of the frame.
    float3 origin = params.cam_orthographic ? eye + U * px + V * py : eye;

    if (params.cam_lens_radius > 0.0f) {
      float u1 = rng.next();
      float u2 = rng.next();
      vibrt_cycles::KernelCamera camera{params.cam_aperture_blades, params.cam_aperture_rotation,
                                        params.cam_inv_aperture_ratio};
      auto bokeh = vibrt_cycles::camera_sample_aperture(&camera, vibrt_cycles::make_float2(u1, u2));
      float lx = bokeh.x * params.cam_lens_radius;
      float ly = bokeh.y * params.cam_lens_radius;
      float3 Uunit = normalize3(U);
      float3 Vunit = normalize3(V);
      // Cycles' focus_distance is measured from the camera (eye),
      // not from the near-clip plane. Place the focus point at
      // eye + dir*focal_distance so the bokeh is sharp where Cycles
      // says it should be.
      float3 focus_point = origin + dir * (params.cam_focal_distance / fmaxf(dot3(dir, W), 1e-6f));
      origin = origin + Uunit * lx + Vunit * ly;
      dir = normalize3(focus_point - origin);
    }

    float transparency;
    float pixel_radius = sqrtf(dot3(U, U)) / dim.x + sqrtf(dot3(V, V)) / dim.y;
    float ray_dP = params.cam_orthographic ? pixel_radius : 0;
    float ray_dD = params.cam_orthographic ? 0 :
        0.5f * (vibrt_cycles::len(cycles_vector(normalize3(U * (px + 2.0f / dim.x) + V * py + W) -
                                               normalize3(U * px + V * py + W))) +
                vibrt_cycles::len(cycles_vector(normalize3(U * px + V * (py + 2.0f / dim.y) + W) -
                                               normalize3(U * px + V * py + W))));
    float3 contribution = trace_path(origin, dir, rng, params.cam_clip_start, transparency, ray_dP, ray_dD);
    accum = accum + contribution;
    if (adaptive && vibrt_adaptive::sample_is_class_A(0, s)) {
      adaptive[4] += 2 * contribution.x;
      adaptive[5] += 2 * contribution.y;
      adaptive[6] += 2 * contribution.z;
    }
    transparency_sum += transparency;
  }
  if (params.accumulation != nullptr) {
    float *sum = &params.accumulation[pixel * 4];
    sum[0] = accum.x;
    sum[1] = accum.y;
    sum[2] = accum.z;
    sum[3] = transparency_sum;
  }
  float inv = 1.0f / (float)sample_end;
  if (adaptive) {
    adaptive[0] = accum.x; adaptive[1] = accum.y; adaptive[2] = accum.z;
    adaptive[3] = transparency_sum;
    adaptive[8] = __uint_as_float(sample_end);
  }
  accum = accum * inv;

  // y is 0 at top of image; we'll flip on CPU side for EXR.
  float *out = &params.image[pixel * 4];
  out[0] = accum.x;
  out[1] = accum.y;
  out[2] = accum.z;
  out[3] = 1.0f - __saturatef(transparency_sum * inv);
}
