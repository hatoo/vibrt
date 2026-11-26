struct SceneQuery {
  HitGroupData *geometry;
  unsigned int hit_primitive;
  bool hit;
  float distance;
  float2 bary;
  float3 P, N, Ng;
};
static __device__ float3 query_inverse_direction(float3 D) {
  // A ray visits many BVH nodes with the same direction. Parallel slabs keep
  // their explicit test below, so they do not need an infinite reciprocal.
  return make_float3(fabsf(D.x) < 1e-30f ? 0.0f : 1.0f / D.x,
                     fabsf(D.y) < 1e-30f ? 0.0f : 1.0f / D.y,
                     fabsf(D.z) < 1e-30f ? 0.0f : 1.0f / D.z);
}
static __device__ bool query_box(const QueryNode &node, float3 P, float3 D,
                                 float3 invD,
                                 float tmin, float tmax) {
  const float *p = &P.x, *d = &D.x, *inv = &invD.x;
  for (int axis = 0; axis < 3; axis++) {
    if (fabsf(d[axis]) < 1e-30f) {
      if (p[axis] < node.lo[axis] || p[axis] > node.hi[axis])
        return false;
    } else {
      float a = (node.lo[axis] - p[axis]) * inv[axis],
            b = (node.hi[axis] - p[axis]) * inv[axis];
      float near = fminf(a, b), far = fmaxf(a, b);
      // Round the interval outwards to cover reciprocal/multiply error at
      // grazing bounds. This only admits extra nodes; triangle tests retain
      // their original arithmetic, distances, and traversal order.
      constexpr float padding = 4.0f * 1.1920928955078125e-7f;
      tmin = fmaxf(tmin, near - fmaxf(fabsf(near), 1e-30f) * padding);
      tmax = fminf(tmax, far + fmaxf(fabsf(far), 1e-30f) * padding);
      if (tmin > tmax)
        return false;
    }
  }
  return true;
}
static __device__ __noinline__ void query_mesh(const QueryObject &obj,
                                               const PathVertex &v, float3 P,
                                               float3 D, float tmin,
                                               bool skip_self, SceneQuery &q) {
  HitGroupData *hg = obj.geometry;
  P = transform_point(hg->world_to_object, P);
  D = transform_vector(hg->world_to_object, D);
  float3 invD = query_inverse_direction(D);
  unsigned int stack[64];
  int sp = 0;
  stack[sp++] = 0;
  while (sp) {
    const QueryNode &node = obj.nodes[stack[--sp]];
    if (!query_box(node, P, D, invD, tmin, q.distance))
      continue;
    if (node.count == 0) {
      stack[sp++] = node.first;
      stack[sp++] = node.first + 1;
      continue;
    }
    for (unsigned int i = 0; i < node.count; i++) {
      unsigned int prim = obj.indices[node.first + i];
      if (skip_self && hg->instance_id == v.geometry->instance_id &&
          prim == v.primitive)
        continue;
      const int *ids = hg->indices + prim * 3;
      float3 A = make_f3(hg->vertices + 3 * ids[0]);
      float3 E1 = make_f3(hg->vertices + 3 * ids[1]) - A,
             E2 = make_f3(hg->vertices + 3 * ids[2]) - A;
      float3 H = cross3(D, E2);
      float det = dot3(E1, H);
      if (fabsf(det) < 1e-20f)
        continue;
      float inv = 1 / det;
      float3 S = P - A;
      float u = dot3(S, H) * inv;
      if (u < 0 || u > 1)
        continue;
      float3 Q = cross3(S, E1);
      float w = dot3(D, Q) * inv;
      if (w < 0 || u + w > 1)
        continue;
      float distance = dot3(E2, Q) * inv;
      if (distance < tmin || distance > q.distance)
        continue;
      q.hit = true;
      q.distance = distance;
      q.geometry = hg;
      q.hit_primitive = prim;
      q.bary = make_float2(u, w);
    }
  }
}
static __device__ __noinline__ SceneQuery
query_scene(const PathVertex &v, float3 origin, float3 direction,
            float distance, bool only_local, bool skip_self,
            unsigned int mask = 0xffu, float tmin = 0.0f) {
  SceneQuery q = {};
  q.distance = distance;
  if (skip_self)
    tmin = fmaxf(tmin, 1e-6f);
  if (distance <= tmin || dot3(direction, direction) < 1e-30f || !v.geometry ||
      !params.query_nodes)
    return q;
  if (only_local) {
    // Query objects retain scene-object order, matching the SBT instance IDs.
    // Local queries need only the entry instance's mesh BVH, even when other
    // instances share that mesh. Keep its visibility mask and self-hit policy.
    const QueryObject &obj = params.query_objects[v.geometry->instance_id];
    if (obj.mask & mask)
      query_mesh(obj, v, origin, direction, tmin, skip_self, q);
  } else {
    float3 invD = query_inverse_direction(direction);
    unsigned int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    while (sp) {
      const QueryNode &node = params.query_nodes[stack[--sp]];
      if (!query_box(node, origin, direction, invD, tmin, q.distance))
        continue;
      if (node.count == 0) {
        stack[sp++] = node.first;
        stack[sp++] = node.first + 1;
        continue;
      }
      for (unsigned int i = 0; i < node.count; i++) {
        const QueryObject &obj =
            params.query_objects[params.query_indices[node.first + i]];
        if (!(obj.mask & mask))
          continue;
        query_mesh(obj, v, origin, direction, tmin, skip_self, q);
      }
    }
  }
  if (q.hit) {
    HitGroupData *hg = q.geometry;
    const int *ids = hg->indices + 3 * q.hit_primitive;
    float3 A = make_f3(hg->vertices + 3 * ids[0]),
           B = make_f3(hg->vertices + 3 * ids[1]),
           C = make_f3(hg->vertices + 3 * ids[2]);
    float3 N = cross3(B - A, C - A);
    q.Ng = normalize3(transform_normal(hg->world_to_object, N));
    if (hg->normals)
      N = make_f3(hg->normals + 3 * ids[0]) * (1 - q.bary.x - q.bary.y) +
          make_f3(hg->normals + 3 * ids[1]) * q.bary.x +
          make_f3(hg->normals + 3 * ids[2]) * q.bary.y;
    q.N = normalize3(transform_normal(hg->world_to_object, N));
    q.P = origin + direction * q.distance;
  }
  return q;
}

static __device__ __noinline__ void initialize_volume_stack(float3 origin, VolumeStack &stack) {
  float distances[VOL_STACK_MAX];
  const float3 direction = normalize3(make_float3(0.371f, 0.529f, 0.763f));
  for (unsigned int i = 0; i < params.num_initial_volumes; i++) {
    const QueryObject &object = params.initial_volumes[i];
    HitGroupData *hg = object.geometry;
    float3 local = transform_point(hg->world_to_object, origin);
    bool in_bounds = true;
    for (int axis = 0; axis < 3; axis++)
      in_bounds &= (&local.x)[axis] >= hg->bounds_min[axis] &&
                   (&local.x)[axis] <= hg->bounds_max[axis];
    if (!in_bounds) continue;
    PathVertex vertex = {};
    SceneQuery hit = {};
    hit.distance = 1e20f;
    query_mesh(object, vertex, origin, direction, 1e-6f, false, hit);
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
  float3 P = v.P;
  for (int step = 0; step < 1024; step++) {
    float probabilities[3], total = 0;
    for (int c = 0; c < 3; c++) {
      probabilities[c] = fabsf((&weight.x)[c]) * alpha[c];
      total += probabilities[c];
    }
    if (!(total > 1e-12f)) return false;
    for (int c = 0; c < 3; c++) probabilities[c] /= total;
    float pick = rng.next();
    int channel = pick < probabilities[0] ? 0 : pick < probabilities[0]+probabilities[1] ? 1 : 2;
    float distance = -logf(fmaxf(1-rng.next(), 1e-7f)) / sigma[channel];
    SceneQuery q = query_scene(v, P, D, distance, true, step == 0, 0xffu, 1e-7f);
    float t = q.hit ? q.distance : distance;
    float tr[3], pdf = 0;
    for (int c = 0; c < 3; c++) {
      tr[c] = expf(-sigma[c]*t);
      pdf += probabilities[c]*tr[c]*(q.hit ? 1 : sigma[c]);
    }
    if (!(pdf > 1e-30f)) return false;
    for (int c = 0; c < 3; c++)
      (&weight.x)[c] *= tr[c]*(q.hit ? 1 : sigma[c]*alpha[c])/pdf;
    if (q.hit) {
      throughput = throughput * weight;
      v.P=q.P;v.Ng=q.Ng*orientation;v.Ns=q.N*orientation;
      v.primitive=q.hit_primitive;v.barycentric=q.bary;
      e.Ng=v.Ng;e.Ns=v.Ns;build_frame(e.Ns,e.T,e.B);
      e.base_color=make_float3(1,1,1);
      return true;
    }
    P = P + D*t;
    if (step >= 3) {
      float survival = fminf(fmaxf(fmaxf(weight.x, weight.y), weight.z), 0.95f);
      if (!(survival > 0) || rng.next() >= survival) return false;
      weight = weight / survival;
    }
    build_frame(D, T, B);
    float phase_pdf;
    float3 local = phase_hg_sample(g, rng.next(), rng.next(), phase_pdf);
    D = normalize3(T*local.x+B*local.y+D*local.z);
  }
  return false;
}

// Cycles' three-axis Burley disk estimator. Scene intersections use vibrt's
// BVH.
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
    ::float3 base = ::cross3(v.dPdu, v.dPdv);
    float sign = ::dot3(base, v.Ns) < 0 ? -1 : 1;
    ::float3 dpdu = v.dPdu + (cuda_vec(input(1)) - cuda_vec(input(0))) * 1000;
    ::float3 dpdv = v.dPdv + (cuda_vec(input(2)) - cuda_vec(input(0))) * 1000;
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
      value.x += du;
      value.y += dv;
      return value;
    }
    ::float3 delta = ::make_float3(0, 0, 0);
    if (index == 1)
      delta = v.dGdu * du + v.dGdv * dv;
    if (index == 2)
      delta = v.dOdu * du + v.dOdv * dv;
    if (index == 3)
      delta = v.dPdu * du + v.dPdv * dv;
    value.x += delta.x;
    value.y += delta.y;
    value.z += delta.z;
    return value;
  }
#endif
#if (VIBRT_SHADER_OPS & (1ull << 47))
  if ((n.op & 255) == 47) {
    ::float3 normal = ::normalize3(cuda_vec(input(5)));
    ::float3 r1 = ::cross3(v.dPdv, normal), r2 = ::cross3(normal, v.dPdu);
    float det = ::dot3(v.dPdu, r1);
    if (fabsf(det) < 1e-12f)
      return result(normal);
    float du = (input(1).x - input(0).x) * 1000,
          dv = (input(2).x - input(0).x) * 1000;
    float distance = input(4).x * (n.modes[0] ? -1 : 1),
          strength = __saturatef(input(3).x);
    ::float3 bumped =
        ::normalize3(normal - (r1 * du + r2 * dv) * (distance / det));
    return result(::normalize3(normal * (1 - strength) + bumped * strength));
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
      t = ::normalize3(t - v.Ns * ::dot3(t, v.Ns));
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
    float size = input(0).x, pixel_width = 1;
    if (n.modes[0]) {
      ::float3 camera_to_P = P - ::make_f3(params.cam_eye);
      float depth = ::dot3(camera_to_P, ::normalize3(::make_f3(params.cam_w)));
      ::float3 dx = ::make_f3(params.cam_u) * (2 * fabsf(depth) / params.width);
      ::float3 dy =
          ::make_f3(params.cam_v) * (2 * fabsf(depth) / params.height);
      dx = dx - v.incoming * ::dot3(dx, v.incoming);
      dy = dy - v.incoming * ::dot3(dy, v.incoming);
      pixel_width = 0.5f * (sqrtf(::dot3(dx, dx)) + sqrtf(::dot3(dy, dy)));
    }
    float w = 0.5f * size * pixel_width;
    ::float3 co[3];
    for (int i = 0; i < 3; i++)
      co[i] = ::transform_point(
          v.geometry->object_to_world,
          ::make_f3(v.geometry->vertices +
                    3 * v.geometry->indices[3 * v.primitive + i]));
    for (int i = 0; i < 3; i++) {
      ::float3 edge = co[i] - co[(i + 2) % 3], area = ::cross3(edge, P - co[i]);
      if (::dot3(area, area) < ::dot3(edge, edge) * w * w)
        return scalar(1);
    }
    return scalar(0);
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
