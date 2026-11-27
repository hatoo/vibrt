struct SceneQuery {
  HitGroupData *geometry;
  unsigned int hit_primitive;
  bool hit;
  float distance;
  float2 bary;
  float3 P, N, Ng;
};
// Raw geometry queries share the radiance/shadow OptiX acceleration structures.
struct QueryPayload {
  SceneQuery *result;
  unsigned int instance, primitive;
  bool only_local, skip_self;
};
static __device__ __noinline__ SceneQuery
query_scene_impl(const PathVertex &v, float3 origin, float3 direction,
            float distance, bool only_local, bool skip_self,
            unsigned int mask = 0xffu, float tmin = 0.0f) {
  SceneQuery q = {};
  q.distance = distance;
  if (skip_self) tmin = fmaxf(tmin, 1e-6f);
  if (distance <= tmin || dot3(direction, direction) < 1e-30f || !v.geometry)
    return q;
  QueryPayload payload = {&q, v.geometry->instance_id, v.primitive, only_local, skip_self};
  unsigned int hi, lo;
  pack_ptr(&payload, hi, lo);
  optixTrace(params.traversable, origin, direction, tmin, distance, 0.0f,
             OptixVisibilityMask(mask), OPTIX_RAY_FLAG_ENFORCE_ANYHIT, 2, 3, 2, hi, lo);
  if (q.hit) {
    HitGroupData *hg = q.geometry;
    const int *ids = hg->indices + 3 * q.hit_primitive;
    float3 A = make_f3(hg->vertices + 3 * ids[0]),
           B = make_f3(hg->vertices + 3 * ids[1]),
           C = make_f3(hg->vertices + 3 * ids[2]);
    float3 N = cross3(B - A, C - A);
    if (hg->curve_radii)
      N = transform_point(hg->world_to_object, origin + direction * q.distance) - (A * (1 - q.bary.x) + B * q.bary.x);
    q.Ng = normalize3(transform_normal(hg->world_to_object, N));
    if (hg->normals && !hg->curve_radii)
      N = make_f3(hg->normals + 3 * ids[0]) * (1 - q.bary.x - q.bary.y) +
          make_f3(hg->normals + 3 * ids[1]) * q.bary.x +
          make_f3(hg->normals + 3 * ids[2]) * q.bary.y;
    q.N = normalize3(transform_normal(hg->world_to_object, N));
    q.P = origin + direction * q.distance;
  }
  return q;
}

// Isolate trace payload analysis from the shader interpreter's stack arrays.
struct QueryCall {
  const PathVertex *vertex;
  float3 origin, direction;
  float distance;
  bool only_local, skip_self;
  unsigned int mask;
  float tmin;
  SceneQuery result;
};
static __device__ SceneQuery query_scene(const PathVertex &v, float3 origin, float3 direction,
                                        float distance, bool only_local, bool skip_self,
                                        unsigned int mask = 0xffu, float tmin = 0.0f) {
  QueryCall call = {&v, origin, direction, distance, only_local, skip_self, mask, tmin, {}};
  optixContinuationCall<void, QueryCall *>(0, &call);
  return call.result;
}

static __device__ __noinline__ void initialize_volume_stack(float3 origin, VolumeStack &stack) {
  float distances[VOL_STACK_MAX];
  const float3 direction = normalize3(make_float3(0.371f, 0.529f, 0.763f));
  for (unsigned int i = 0; i < params.num_initial_volumes; i++) {
    HitGroupData *hg = params.initial_volumes[i];
    float3 local = transform_point(hg->world_to_object, origin);
    bool in_bounds = true;
    for (int axis = 0; axis < 3; axis++)
      in_bounds &= (&local.x)[axis] >= hg->bounds_min[axis] &&
                   (&local.x)[axis] <= hg->bounds_max[axis];
    if (!in_bounds) continue;
    PathVertex vertex = {};
    vertex.geometry = hg;
    SceneQuery hit = query_scene(vertex, origin, direction, 1e20f, true, false, 0xffu, 1e-6f);
    if (!hit.hit) continue;
    const int *ids = hg->indices + hit.hit_primitive * 3;
    float3 a = make_f3(hg->vertices + ids[0]*3);
    float3 n = cross3(make_f3(hg->vertices + ids[1]*3)-a,
                     make_f3(hg->vertices + ids[2]*3)-a);
    n = transform_normal(hg->world_to_object, n);
    // The closest crossing is outward exactly when the origin is inside.
    if (dot3(n, direction) <= 0) continue;
    PrincipledGpu *material = hg->mat;
    if (hg->material_indices && hg->num_materials) {
      unsigned int index = hg->material_indices[hit.hit_primitive];
      if (index < hg->num_materials) material = hg->materials[index];
    }
    if (!material || !material->volume || stack.depth == VOL_STACK_MAX) continue;
    int pos = stack.depth++;
    // Outer media first, nearest exit (innermost volume) at the top.
    while (pos > 0 && distances[pos-1] < hit.distance) {
      distances[pos] = distances[pos-1];
      stack.entries[pos] = stack.entries[pos-1];
      pos--;
    }
    distances[pos] = hit.distance;
    stack.entries[pos] = material->volume;
  }
}

#if (VIBRT_CLOSURE_TYPES & (1u << 7))
// Random walk uses Cycles' albedo-to-extinction fit. Geometry queries stay
// inside the entry object; spectral free flights use a matched RGB mixture.
static __device__ __noinline__ bool scatter_subsurface_walk(
    PathVertex &v, MaterialEval &e, RNG &rng, float3 &throughput) {
  float radius[3] = {e.sss_radius.x, e.sss_radius.y, e.sss_radius.z};
  float albedo[3] = {e.base_color.x, e.base_color.y, e.base_color.z};
  float sigma[3], alpha[3];
  float g = fminf(fmaxf(e.sss_anisotropy, 0.0f), 0.9f);
  float ior = e.mat->closure_type == 7 || e.mat->sss_method == 3 ? e.sss_ior : e.ior;
  ior = fminf(fmaxf(ior, 1.01f), 3.8f);
  float3 weight = make_float3(1, 1, 1);
  for (int c = 0; c < 3; c++) {
    albedo[c] = __saturatef(albedo[c]);
    float d = fmaxf(radius[c] * e.sss_scale, 1e-6f);
    if (e.mat->sss_method == 3) {
      float inv = 1 / ior;
      float f = inv * (-1.440f * inv + 0.710f) + 0.668f + 0.0636f * ior;
      float a = vibrt_cycles::bssrdf_dipole_compute_alpha_prime(
          albedo[c], (4.0f / 3.0f) * (1 + f) / (1 - f));
      d *= sqrtf(3 * (1 - a));
    } else {
      d *= 0.25f * INV_PIf;
    }
    vibrt_cycles::subsurface_random_walk_remap(albedo[c], d, g, &sigma[c], &alpha[c]);
    if (alpha[c] < 0.2f) {
      (&weight.x)[c] *= alpha[c] / 0.2f;
      alpha[c] = 0.2f;
    }
  }
  float3 T, B, N = e.Ns;
  build_frame(N, T, B);
  float u = rng.next(), w = rng.next();
  float3 D;
  if (e.mat->sss_method == 3 && u < 0.5f) {
    float r = sqrtf(2 * u), phi = 2 * M_PIf * w;
    D = T * (r * cosf(phi)) + B * (r * sinf(phi)) - N * sqrtf(1 - 2 * u);
  } else {
    if (e.mat->sss_method == 3) u = 2 * u - 1;
    float a = e.mat->sss_method == 3 ? 1.0f : e.alpha;
    float3 I = v.incoming;
    float3 H = sample_ggx_vndf_aniso(
        make_float3(dot3(I,T),dot3(I,B),dot3(I,N)), a, a, u, w);
    H = normalize3(T*H.x+B*H.y+N*H.z);
    float cosine = dot3(I,H), eta = 1 / ior;
    D = -I * eta + H * (eta*cosine-sqrtf(fmaxf(1-eta*eta*(1-cosine*cosine),0)));
  }
  float orientation = dot3(v.Ng, N) >= 0 ? 1.0f : -1.0f;
  if (dot3(v.Ng * orientation, D) >= 0) return false;
  // Cycles' Dwivedi guiding uses a mixture with the original HG walk. Keep
  // all direction/distance/channel probabilities in the same balance PDF.
  const float3 entry = v.P;
  float3 P = entry;
  float diffusion =
      vibrt_cycles::diffusion_length_dwivedi(fmaxf(alpha[0], fmaxf(alpha[1], alpha[2])));
  float phase_log =
      diffusion > 1 && isfinite(diffusion) ? logf((diffusion + 1) / (diffusion - 1)) : 0;
  float guide_fraction = phase_log > 0 ? 1 - fmaxf(0.5f, powf(fabsf(g), 0.125f)) : 0;
  bool have_opposite = false;
  float opposite_distance = 0;
  auto extinction = vibrt_cycles::make_float3(sigma[0], sigma[1], sigma[2]);
  for (int step = 0; step < 1024; step++) {
    float probabilities[3], total = 0;
    for (int c = 0; c < 3; c++) {
      probabilities[c] = fabsf((&weight.x)[c]) * alpha[c];
      total += probabilities[c];
    }
    if (!(total > 1e-12f))
      return false;
    for (int c = 0; c < 3; c++)
      probabilities[c] /= total;
    float pick = rng.next();
    int channel = pick < probabilities[0] ? 0 : pick < probabilities[0] + probabilities[1] ? 1 : 2;
    float sample_sigma = sigma[channel];
    float backward_fraction = 0, forward_factor = 0, backward_factor = 0;
    float forward_stretch = 1, backward_stretch = 1;
    if (step > 0) {
      bool guided = rng.next() < guide_fraction;
      bool backward = false;
      if (have_opposite && guide_fraction > 0) {
        float x = fminf(fmaxf(dot3(P - entry, -N), 0), opposite_distance);
        backward_fraction = 1 / (1 + expf((opposite_distance - 2 * x) / diffusion));
        backward = rng.next() < backward_fraction;
      }
      float u1 = rng.next(), u2 = rng.next(), hg_pdf, cosine;
      if (guided) {
        cosine = vibrt_cycles::sample_phase_dwivedi(diffusion, phase_log, u1);
        if (backward)
          cosine = -cosine;
        build_frame(N, T, B);
        float r = sqrtf(fmaxf(0, 1 - cosine * cosine)), phi = 2 * M_PIf * u2;
        float3 next = N * cosine + (T * cosf(phi) + B * sinf(phi)) * r;
        hg_pdf = phase_hg_eval(g, dot3(D, next));
        D = next;
      } else {
        build_frame(D, T, B);
        float3 local = phase_hg_sample(g, u1, u2, hg_pdf);
        D = normalize3(T * local.x + B * local.y + D * local.z);
        cosine = dot3(D, N);
      }
      if (guide_fraction > 0) {
        forward_factor = 0.5f * INV_PIf *
                         vibrt_cycles::eval_phase_dwivedi(diffusion, phase_log, cosine) / hg_pdf;
        backward_factor = 0.5f * INV_PIf *
                          vibrt_cycles::eval_phase_dwivedi(diffusion, phase_log, -cosine) / hg_pdf;
        forward_stretch = 1 - cosine / diffusion;
        backward_stretch = 1 + cosine / diffusion;
        if (guided)
          sample_sigma *= backward ? backward_stretch : forward_stretch;
      }
    }
    float distance = -logf(fmaxf(1 - rng.next(), 1e-7f)) / sample_sigma;
    float query_distance =
        step == 0 ? fmaxf(distance, 10 / fminf(sigma[0], fminf(sigma[1], sigma[2]))) : distance;
    SceneQuery q = query_scene(v, P, D, query_distance, true, step == 0, 0xffu, 1e-7f);
    if (step == 0 && q.hit) {
      opposite_distance = dot3(q.P - entry, -N);
      have_opposite = opposite_distance > 0;
    }
    bool hit = q.hit && q.distance < distance;
    float t = hit ? q.distance : distance;
    vibrt_cycles::Spectrum tr;
    auto density = vibrt_cycles::subsurface_random_walk_pdf(extinction, t, hit, &tr);
    if (step > 0 && guide_fraction > 0) {
      auto forward =
          vibrt_cycles::subsurface_random_walk_pdf(extinction * forward_stretch, t, hit, nullptr) *
          forward_factor;
      auto backward =
          vibrt_cycles::subsurface_random_walk_pdf(extinction * backward_stretch, t, hit, nullptr) *
          backward_factor;
      auto guided = forward * (1 - backward_fraction) + backward * backward_fraction;
      density = density * (1 - guide_fraction) + guided * guide_fraction;
    }
    float pdf =
        probabilities[0] * density.x + probabilities[1] * density.y + probabilities[2] * density.z;
    if (!(pdf > 1e-30f))
      return false;
    for (int c = 0; c < 3; c++)
      (&weight.x)[c] *= tr[c] * (hit ? 1 : sigma[c] * alpha[c]) / pdf;
    if (hit) {
      throughput = throughput * weight;
      v.P=q.P;v.Ng=q.Ng*orientation;v.Ns=q.N*orientation;
      v.primitive=q.hit_primitive;v.barycentric=q.bary;
      e.Ng=v.Ng;e.Ns=v.Ns;build_frame(e.Ns,e.T,e.B);
      e.base_color=make_float3(1,1,1);
      return true;
    }
    P = P + D * t;
    if (step >= 3) {
      float survival = fminf(fmaxf(fmaxf(weight.x, weight.y), weight.z), 0.95f);
      if (!(survival > 0) || rng.next() >= survival)
        return false;
      weight = weight / survival;
    }
  }
  return false;
}

// Cycles' three-axis Burley disk estimator with OptiX geometry queries.
static __device__ __noinline__ bool scatter_subsurface(PathVertex &v,
                                                       MaterialEval &e,
                                                       RNG &rng,
                                                       float3 &throughput) {
  auto radius =
      vibrt_cycles::make_float3(e.sss_radius.x, e.sss_radius.y, e.sss_radius.z);
  float colors[3] = {e.base_color.x, e.base_color.y, e.base_color.z};
  for (int c = 0; c < 3; c++) {
    float a = __saturatef(colors[c]);
    radius[c] = fmaxf(radius[c] * e.sss_scale, 0) * (0.25f * INV_PIf) /
                (1.9f - a + 3.5f * (a - 0.8f) * (a - 0.8f));
  }
  if (vibrt_cycles::reduce_max(radius) < 1e-8f)
    return true;
  float3 N = v.Ng, T, B;
  build_frame(N, T, B);
  float probabilities[3] = {0.5f, 0.25f, 0.25f};
  float axis = rng.next();
  if (axis >= 0.5f && axis < 0.75f) {
    float3 old = N;
    N = T;
    T = old;
    probabilities[0] = 0.25f;
    probabilities[1] = 0.5f;
  } else if (axis >= 0.75f) {
    float3 old = N;
    N = B;
    B = old;
    probabilities[0] = 0.25f;
    probabilities[2] = 0.5f;
  }
  float r, h;
  vibrt_cycles::bssrdf_sample(radius, rng.next(), &r, &h);
  float pdf = vibrt_cycles::bssrdf_pdf(radius, r);
  if (!(pdf > 0) || !(h > 0))
    return false;
  float phi = 2 * M_PIf * rng.next();
  float3 origin = v.P + N * h + (T * cosf(phi) + B * sinf(phi)) * r;
  float tmin = 0, sum = 0, chosen_energy = 0;
  SceneQuery chosen = {};
  float3 chosen_weight = make_float3(0, 0, 0);
  for (int i = 0; i < 64; i++) {
    SceneQuery q = query_scene(v, origin, -N, 2 * h, true, false, 0xffu, tmin);
    if (!q.hit)
      break;
    tmin = nextafterf(q.distance, 1e30f);
    float pN = probabilities[0] * fabsf(dot3(N, q.Ng)),
          pT = probabilities[1] * fabsf(dot3(T, q.Ng)),
          pB = probabilities[2] * fabsf(dot3(B, q.Ng));
    float denom = pN * pN + pT * pT + pB * pB;
    if (denom < 1e-20f)
      continue;
    auto profile =
        vibrt_cycles::bssrdf_eval(radius, sqrtf(dot3(q.P - v.P, q.P - v.P)));
    float3 weight =
        make_float3(profile.x, profile.y, profile.z) * (pN / (denom * pdf));
    float energy = (weight.x + weight.y + weight.z) / 3;
    sum += energy;
    if (energy > 0 && rng.next() * sum < energy) {
      chosen = q;
      chosen_weight = weight;
      chosen_energy = energy;
    }
  }
  if (!chosen.hit || chosen_energy <= 0)
    return false;
  throughput = throughput * chosen_weight * (sum / chosen_energy);
  v.P = chosen.P;
  v.Ng = chosen.Ng;
  v.Ns = chosen.N;
  v.primitive = chosen.hit_primitive;
  v.barycentric = chosen.bary;
  e.Ns = v.Ns;
  e.Ng = v.Ng;
  build_frame(e.Ns, e.T, e.B);
  return true;
}
#endif

#if (VIBRT_SHADER_OPS & ((255ull << 40) | (1ull << 51)))
namespace vibrt_shader {
VN_INLINE vibrt_scene_nodes::ShaderData normal_map_data(const float4 *ctx, float4 tangent,
                                                        bool named, const SceneServices &services) {
  const PathVertex &v = *(const PathVertex *)services.vertex;
  vibrt_scene_nodes::ShaderData sd{};
  bool back = ctx[16].x != 0;
  sd.N = ::cycles_vector(back ? -v.Ns : v.Ns);
  sd.Ng = ::cycles_vector(back ? -v.Ng : v.Ng);
  sd.flag = back ? vibrt_scene_nodes::SD_BACKFACING : 0;
  sd.normal = ::cycles_vector(v.Ns);
  sd.tangent = xyz(tangent);
  sd.sign = tangent.w;
  sd.object_to_world = ctx + 22;
  sd.world_to_object = ctx + 25;
  if (v.geometry) {
    if (v.geometry->curve_radii) sd.type = vibrt_scene_nodes::PRIMITIVE_CURVE;
    sd.world_basis = false;
    sd.tangent = vibrt_scene_nodes::matrix_direction(sd.world_to_object, sd.tangent);
    if (!named)
      sd.tangent = safe_normalize(sd.tangent);
    if (v.geometry->normals) {
      sd.normal = make_float3(0);
      const int *indices = v.geometry->indices + 3 * v.primitive;
      float weights[3] = {1 - v.barycentric.x - v.barycentric.y, v.barycentric.x, v.barycentric.y};
      for (int i = 0; i < 3; ++i)
        sd.normal += ::cycles_vector(::make_f3(v.geometry->normals + 3 * indices[i])) * weights[i];
    }
    else
      sd.shader = 0;
  }
  return sd;
}
#if (VIBRT_SHADER_OPS & (7ull << 41))
VN_INLINE float4 tracing_node(const ShaderInstruction &, const float4 *,
                              const float4 *, const SceneServices &);
#endif
VN_INLINE float4 scene_node(const ShaderInstruction &n, const float4 *slots,
                            const float4 *ctx, const SceneServices &services) {
  const PathVertex &v = *(const PathVertex *)services.vertex;
  auto input = [&](int i) { return slots[n.inputs[i]]; };
  auto cuda_vec = [](float4 a) { return ::make_float3(a.x, a.y, a.z); };
  auto result = [](::float3 a) { return make_float4(a.x, a.y, a.z, 1); };
#if (VIBRT_SHADER_OPS & (1ull << 51))
  if ((n.op & 255) == 51) {
    auto dP = vibrt_differential::differential_from_compact(::cycles_vector(v.Ng), v.dP);
    ::float3 dx = ::make_float3(dP.dx.x, dP.dx.y, dP.dx.z);
    ::float3 dy = ::make_float3(dP.dy.x, dP.dy.y, dP.dy.z);
    ::float3 base = ::cross3(dx, dy);
    float sign = ::dot3(base, v.Ns) < 0 ? -1 : 1;
    ::float3 dpdu = dx + (cuda_vec(input(1)) - cuda_vec(input(0))) * 1000;
    ::float3 dpdv = dy + (cuda_vec(input(2)) - cuda_vec(input(0))) * 1000;
    ::float3 changed = ::normalize3(::cross3(dpdu, dpdv)) * sign;
    if (::dot3(changed, changed) < 1e-12f)
      return input(3);
    return result(
        ::normalize3(cuda_vec(input(3)) + changed - ::normalize3(base) * sign));
  }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 46))
  if ((n.op & 255) == 46) {
    unsigned int index = n.modes[0];
    float4 value = ctx[index];
    float du = n.params[0], dv = n.params[1];
    if (index == 0) {
      value.x += v.du.dx * du + v.du.dy * dv;
      value.y += v.dv.dx * du + v.dv.dy * dv;
      return value;
    }
    auto dP = vibrt_differential::differential_from_compact(::cycles_vector(v.Ng), v.dP);
    ::float3 delta = ::make_float3(0, 0, 0);
    float su = v.du.dx * du + v.du.dy * dv, sv = v.dv.dx * du + v.dv.dy * dv;
    if (index == 1)
      delta = v.dGdu * su + v.dGdv * sv;
    if (index == 2)
      delta = v.dOdu * su + v.dOdv * sv;
    if (index == 3)
      delta = ::make_float3(dP.dx.x, dP.dx.y, dP.dx.z) * du + ::make_float3(dP.dy.x, dP.dy.y, dP.dy.z) * dv;
    value.x += delta.x;
    value.y += delta.y;
    value.z += delta.z;
    return value;
  }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 47))
  if ((n.op & 255) == 47) {
    vibrt_scene_nodes::ShaderData sd{};
    sd.N = ::cycles_vector(::normalize3(cuda_vec(input(5))));
    sd.Ng = ::cycles_vector(v.Ng);
    sd.world_basis = !n.modes[1];
    float4 object_to_world[3], world_to_object[3];
    if (v.geometry) {
      for (int i = 0; i < 3; ++i) {
        const float *a = v.geometry->object_to_world + 4 * i;
        const float *b = v.geometry->world_to_object + 4 * i;
        object_to_world[i] = make_float4(a[0], a[1], a[2], a[3]);
        world_to_object[i] = make_float4(b[0], b[1], b[2], b[3]);
      }
      sd.object_to_world = object_to_world;
      sd.world_to_object = world_to_object;
    }
    auto dP = vibrt_differential::differential_from_compact(sd.Ng, v.dP);
    return color(vibrt_scene_nodes::bump(sd, dP, input(0).x, input(1).x, input(2).x,
                                         input(3).x, input(4).x, n.params[0] > 0 ? n.params[0] : 0.001f,
                                         n.modes[0] != 0, n.modes[1] != 0));
  }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 45))
  if ((n.op & 255) == 45)
    return v.geometry && v.geometry->tangents
               ? result(
                     ::normalize3(v.incoming - v.T * ::dot3(v.incoming, v.T)))
               : color(make_float3(0));
#endif
#if (VIBRT_SHADER_OPS & (1ull << 44))
  if ((n.op & 255) == 44) {
    float4 value = make_float4(0);
    bool found = false;
    if (v.geometry)
      for (unsigned int i = 0; i < v.geometry->num_attributes; i++) {
        const ShaderAttribute &a = v.geometry->attributes[i];
        if (a.key != n.data)
          continue;
        found = true;
        if (a.count == 1)
          value = make_float4(a.data[0], a.data[1], a.data[2], a.data[3]);
        else {
          float w[3] = {1 - v.barycentric.x - v.barycentric.y, v.barycentric.x,
                        v.barycentric.y};
          float du = n.params[0] + (services.derivative == 1 ? 0.001f : 0);
          float dv = n.params[1] + (services.derivative == 2 ? 0.001f : 0);
          float screen_u = du, screen_v = dv;
          du = v.du.dx * screen_u + v.du.dy * screen_v;
          dv = v.dv.dx * screen_u + v.dv.dy * screen_v;
          if ((du != 0 || dv != 0) && v.geometry->uvs) {
            const int *indices = v.geometry->indices + 3 * v.primitive;
            const float *uv = v.geometry->uvs;
            float ux = uv[2 * indices[1]] - uv[2 * indices[0]],
                  uy = uv[2 * indices[2]] - uv[2 * indices[0]];
            float vx = uv[2 * indices[1] + 1] - uv[2 * indices[0] + 1],
                  vy = uv[2 * indices[2] + 1] - uv[2 * indices[0] + 1];
            float det = ux * vy - uy * vx;
            if (fabsf(det) > 1e-12f) {
              float db1 = (du * vy - dv * uy) / det,
                    db2 = (dv * ux - du * vx) / det;
              w[0] -= db1 + db2;
              w[1] += db1;
              w[2] += db2;
            }
          }
          for (int j = 0; j < 3; j++) {
            const float *p =
                a.data + 4 * v.geometry->indices[v.primitive * 3 + j];
            value = value + make_float4(p[0], p[1], p[2], p[3]) * w[j];
          }
        }
        break;
      }
    if (n.modes[0] == 4) {
      ::float3 t =
          ::transform_vector(v.geometry->object_to_world, cuda_vec(value));
      return make_float4(t.x, t.y, t.z, value.w);
    }
    if (n.modes[0] == 3)
      return scalar(found ? hash_uint2_to_float((unsigned int)value.x, 0) : 0);
    return n.modes[0] == 1   ? scalar((value.x + value.y + value.z) / 3)
           : n.modes[0] == 2 ? scalar(value.w)
                             : value;
  }
#endif
  ::float3 P = cuda_vec(ctx[3]);
  unsigned int op = n.op & 255;
#if (VIBRT_SHADER_OPS & (1ull << 40))
  if (op == 40) {
    if (!v.geometry)
      return scalar(0);
    vibrt_scene_nodes::ShaderData sd{};
    sd.P = ::cycles_vector(P);
    if (v.geometry->curve_radii) sd.type = vibrt_scene_nodes::PRIMITIVE_CURVE;
    sd.wi = ::cycles_vector(v.incoming);
    vibrt_scene_nodes::SceneData data{};
    for (int i = 0; i < 3; ++i)
      data.vertices[i] = ::cycles_vector(::transform_point(
          v.geometry->object_to_world,
          ::make_f3(v.geometry->vertices + 3 * v.geometry->indices[3 * v.primitive + i])));
    auto dP = vibrt_differential::differential_from_compact(::cycles_vector(v.Ng), v.dP);
    return scalar(vibrt_scene_nodes::wireframe(&data, &sd, dP, input(0).x, n.modes[0], &sd.P));
  }
#endif
#if (VIBRT_SHADER_OPS & (7ull << 41))
  return tracing_node(n, slots, ctx, services);
#else
  return scalar(0);
#endif
}
#if (VIBRT_SHADER_OPS & (7ull << 41))
VN_INLINE float4 tracing_node(const ShaderInstruction &n, const float4 *slots,
                              const float4 *ctx,
                              const SceneServices &services) {
  const PathVertex &v = *(const PathVertex *)services.vertex;
  auto input = [&](int i) { return slots[n.inputs[i]]; };
  auto cuda_vec = [](float4 a) { return ::make_float3(a.x, a.y, a.z); };
  auto result = [](::float3 a) { return make_float4(a.x, a.y, a.z, 1); };
  ::float3 P = cuda_vec(ctx[3]);
  unsigned int op = n.op & 255;
  RNG rng(v.shader_seed, n.inputs[0], op);
#if (VIBRT_SHADER_OPS & (1ull << 41))
  if (op == 41) {
    float distance = input(1).x, ao = 1;
    ::float3 N = ::normalize3(cuda_vec(input(2))), T, B;
    if (n.modes[1])
      N = -N;
    ::build_frame(N, T, B);
    int samples = (int)n.modes[3];
    if (distance > 0 && samples > 0 && v.geometry) {
      int clear = 0;
      for (int i = 0; i < samples; i++) {
        float u = rng.next(), phi = 2 * M_PIf * rng.next(), r = sqrtf(u);
        ::float3 D =
            T * (r * cosf(phi)) + B * (r * sinf(phi)) + N * sqrtf(1 - u);
        if (!::query_scene(v, P, D, distance, n.modes[2] != 0, true, 0x02u).hit)
          clear++;
      }
      ao = float(clear) / samples;
    }
    return n.modes[0] ? make_float4(input(0).x * ao, input(0).y * ao,
                                    input(0).z * ao, 1)
                      : scalar(ao);
  }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 43))
  if (op == 43) {
    ::float3 origin = cuda_vec(input(0)), direction = cuda_vec(input(1));
    float distance = input(2).x;
    bool skip = origin.x == v.P.x && origin.y == v.P.y && origin.z == v.P.z;
    SceneQuery q =
        ::query_scene(v, origin, direction, distance, n.modes[1] != 0, skip);
    switch (n.modes[0]) {
    case 0:
      return scalar(q.hit ? 1 : 0);
    case 1:
      return scalar(
          q.hit && q.geometry->instance_id == v.geometry->instance_id ? 1 : 0);
    case 2:
      return scalar(q.hit ? q.distance : distance);
    case 3:
      return q.hit ? result(q.P) : color(make_float3(0));
    default:
      return q.hit ? result(::dot3(q.N, direction) > 0 ? -q.N : q.N)
                   : color(make_float3(0));
    }
  }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 42))
  // Cycles' three-axis cubic BSSRDF sampling used by the Bevel node.
  // Walk all same-object intersections instead of evaluating their materials.
  float radius = input(0).x;
  ::float3 normal = cuda_vec(input(1));
  int samples = (int)n.modes[0];
  if (radius <= 0 || samples < 1 || !v.geometry)
    return result(normal);
  ::float3 T, B, Ng = v.Ng;
  ::build_frame(Ng, T, B);
  ::float3 sum = ::make_float3(0, 0, 0);
  for (int i = 0; i < samples; i++) {
    float axis_u = rng.next(), u = rng.next();
    ::float3 axis = Ng, dx = T, dy = B;
    float axis_weight = 0.5f, tx_weight = 0.25f, ty_weight = 0.25f;
    if (axis_u < 0.5f)
      axis_u *= 2;
    else if (axis_u < 0.75f) {
      axis = T;
      dx = Ng;
      axis_weight = 0.25f;
      tx_weight = 0.5f;
      axis_u = (axis_u - 0.5f) * 4;
    } else {
      axis = B;
      dy = Ng;
      axis_weight = 0.25f;
      ty_weight = 0.5f;
      axis_u = (axis_u - 0.75f) * 4;
    }
    float r = radius * bevel_cubic_radius(u),
          h = sqrtf(fmaxf(radius * radius - r * r, 0.0f));
    float phi = 2 * M_PIf * axis_u;
    ::float3 origin = P + axis * h + (dx * cosf(phi) + dy * sinf(phi)) * r;
    float tmin = 0;
// Repeated closest hits have no reservoir bias. The cap bounds pathological
// coincident/non-manifold meshes; ordinary closed surfaces need two hits.
#pragma unroll 1
    for (int hit = 0; hit < 64; hit++) {
      SceneQuery q =
          ::query_scene(v, origin, -axis, 2 * h, true, false, 0xffu, tmin);
      if (!q.hit)
        break;
      float pn = axis_weight * fabsf(::dot3(axis, q.Ng));
      float pt = tx_weight * fabsf(::dot3(dx, q.Ng)),
            pb = ty_weight * fabsf(::dot3(dy, q.Ng));
      float weight = pn / fmaxf(pn * pn + pt * pt + pb * pb, 1e-20f);
      ::float3 offset = q.P - P;
      weight *= bevel_cubic_pdf(radius, sqrtf(::dot3(offset, offset))) /
                fmaxf(bevel_cubic_pdf(radius, r), 1e-20f);
      sum = sum + q.N * weight;
      tmin = nextafterf(q.distance, FLT_MAX);
    }
  }
  ::float3 bevel = ::dot3(sum, sum) > 1e-20f ? ::normalize3(sum) : v.Ns;
  return result(::normalize3(normal + bevel - v.Ns));
#endif
  return scalar(0);
}
#endif
} // namespace vibrt_shader
#endif
