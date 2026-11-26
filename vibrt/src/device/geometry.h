// ---------- Closesthit / Miss / Raygen ----------

static __device__ float3 transform_vector(const float *m, float3 v) {
  return make_float3(m[0]*v.x+m[1]*v.y+m[2]*v.z,m[4]*v.x+m[5]*v.y+m[6]*v.z,m[8]*v.x+m[9]*v.y+m[10]*v.z);
}
static __device__ float3 transform_point(const float *m, float3 v) {
  return transform_vector(m,v)+make_float3(m[3],m[7],m[11]);
}
static __device__ float3 transform_normal(const float *inverse, float3 v) {
  return make_float3(inverse[0]*v.x+inverse[4]*v.y+inverse[8]*v.z,inverse[1]*v.x+inverse[5]*v.y+inverse[9]*v.z,inverse[2]*v.x+inverse[6]*v.y+inverse[10]*v.z);
}
static __device__ __noinline__ void fill_surface_vertex(PathVertex *v, HitGroupData *hg, unsigned int prim, float2 bary, float3 incoming, float distance, unsigned int flags) {
  v->mat = hg->mat;
  if (hg->material_indices != nullptr && hg->num_materials > 0) {
    unsigned int mi = hg->material_indices[prim];
    if ((int)mi < hg->num_materials)
      v->mat = hg->materials[mi];
  }
  unsigned int contexts = v->mat ? v->mat->surface_contexts : 0;
  bool bump = v->mat && v->mat->surface_bump != 0;
  bool derivatives = bump || (contexts & ((1u << 7) | (1u << 17)));
  v->light_pdf = hg->light_pdf ? hg->light_pdf[prim] : 0.0f;
  bool mesh_light = hg->area_light_group >= 0 && v->light_pdf > 0.0f;
  int i0 = hg->indices[prim * 3 + 0];
  int i1 = hg->indices[prim * 3 + 1];
  int i2 = hg->indices[prim * 3 + 2];
  float3 p0 = make_f3(&hg->vertices[i0 * 3]);
  float3 p1 = make_f3(&hg->vertices[i1 * 3]);
  float3 p2 = make_f3(&hg->vertices[i2 * 3]);

  float b0 = 1.0f - bary.x - bary.y;
  float3 P_local = p0 * b0 + p1 * bary.x + p2 * bary.y;

  float3 P = transform_point(hg->object_to_world, P_local);

  // Geometric normal in object space
  float3 e1 = p1 - p0;
  float3 e2 = p2 - p0;
  float3 Ng_local = normalize3(cross3(e1, e2));
  float3 Ng = normalize3(transform_normal(hg->world_to_object, Ng_local));

  // Only mesh-light MIS and UV derivatives need world-space triangle edges.
  // Keep the same arithmetic as light sampling, including instance scaling.
  float3 e1_w = make_float3(0,0,0), e2_w = make_float3(0,0,0);
  float prim_area = 0.0f;
  if (mesh_light || (derivatives && hg->uvs != nullptr)) {
    float3 p0_w = transform_point(hg->object_to_world, p0);
    e1_w = transform_point(hg->object_to_world, p1) - p0_w;
    e2_w = transform_point(hg->object_to_world, p2) - p0_w;
    if (mesh_light) {
      float3 ng_w = cross3(e1_w, e2_w);
      prim_area = 0.5f * sqrtf(dot3(ng_w, ng_w));
    }
  }

  float3 Ns = Ng;
  if (hg->normals != nullptr) {
    float3 n0 = make_f3(&hg->normals[i0 * 3]);
    float3 n1 = make_f3(&hg->normals[i1 * 3]);
    float3 n2 = make_f3(&hg->normals[i2 * 3]);
    float3 Ns_local = normalize3(n0 * b0 + n1 * bary.x + n2 * bary.y);
    Ns = normalize3(transform_normal(hg->world_to_object, Ns_local));
  }

  float2 uv = make_float2(0.0f, 0.0f);
  float3 dpdu = make_float3(0,0,0), dpdv = make_float3(0,0,0);
  if (derivatives) build_frame(Ns,dpdu,dpdv);
  if (hg->uvs != nullptr) {
    float2 uv0 = make_float2(hg->uvs[i0 * 2 + 0], hg->uvs[i0 * 2 + 1]);
    float2 uv1 = make_float2(hg->uvs[i1 * 2 + 0], hg->uvs[i1 * 2 + 1]);
    float2 uv2 = make_float2(hg->uvs[i2 * 2 + 0], hg->uvs[i2 * 2 + 1]);
    uv.x = b0 * uv0.x + bary.x * uv1.x + bary.y * uv2.x;
    uv.y = b0 * uv0.y + bary.x * uv1.y + bary.y * uv2.y;
    if (derivatives) {
      float du1=uv1.x-uv0.x,dv1=uv1.y-uv0.y,du2=uv2.x-uv0.x,dv2=uv2.y-uv0.y;
      float det=du1*dv2-dv1*du2;
      if(fabsf(det)>1e-12f){dpdu=(e1_w*dv2-e2_w*dv1)*(1.0f/det);dpdv=(e2_w*du1-e1_w*du2)*(1.0f/det);}
    }
  }

  // Authored per-vertex tangent (currently only emitted for hair ribbons).
  // Read in object space so the strand axis sits in the same frame as the
  // ribbon vertices, then transform alongside the normal. Zero means "no
  // authored tangent" — eval_material falls back to build_frame() in that
  // case, preserving the shader's existing isotropic behaviour.
  float3 T = make_float3(0.0f, 0.0f, 0.0f);
  if (hg->tangents != nullptr) {
    float3 t0 = make_f3(&hg->tangents[i0 * 3]);
    float3 t1 = make_f3(&hg->tangents[i1 * 3]);
    float3 t2 = make_f3(&hg->tangents[i2 * 3]);
    float3 T_local = t0 * b0 + t1 * bary.x + t2 * bary.y;
    if (dot3(T_local, T_local) > 1e-12f) {
      float3 Tw = transform_vector(hg->object_to_world, T_local);
      float Tlen2 = dot3(Tw, Tw);
      if (Tlen2 > 1e-12f) {
        T = Tw * (1.0f / sqrtf(Tlen2));
      }
    }
  }

  float3 vc = make_float3(1.0f, 1.0f, 1.0f);
  if ((contexts & (1u << 8)) && hg->vertex_colors != nullptr) {
    float3 c0 = make_f3(&hg->vertex_colors[i0 * 3]);
    float3 c1 = make_f3(&hg->vertex_colors[i1 * 3]);
    float3 c2 = make_f3(&hg->vertex_colors[i2 * 3]);
    vc = c0 * b0 + c1 * bary.x + c2 * bary.y;
  }

  v->P = P;
  v->primitive = prim;
  v->barycentric = bary;
  v->shader_seed = optixGetLaunchIndex().x +
                   optixGetLaunchIndex().y * params.width +
                   params.sample_start * 1999u;
  v->geometry=hg;v->ray_length=distance;
  v->path_depth=0;v->diffuse_depth=0;v->glossy_depth=0;v->transmission_depth=0;
  v->path_flags=flags;
  v->P_object=P_local;
  float3 lo=make_f3(hg->bounds_min),hi=make_f3(hg->bounds_max);
  v->generated = make_float3(0,0,0);
  if (contexts & (1u << 1))
    v->generated=make_float3(hi.x>lo.x?(P_local.x-lo.x)/(hi.x-lo.x):0.5f,hi.y>lo.y?(P_local.y-lo.y)/(hi.y-lo.y):0.5f,hi.z>lo.z?(P_local.z-lo.z)/(hi.z-lo.z):0.5f);
  v->N_object = (contexts & (1u << 13)) ? normalize3(transform_normal(hg->object_to_world, Ns)) : make_float3(0,0,0);
  v->object_location = make_float3(hg->object_to_world[3],hg->object_to_world[7],hg->object_to_world[11]);
  v->incoming=incoming;
  v->dPdu=dpdu;v->dPdv=dpdv;
  v->dOdu=v->dOdv=v->dGdu=v->dGdv=make_float3(0,0,0);
  if (bump && (contexts & ((1u << 1) | (1u << 2)))) {
    v->dOdu=transform_vector(hg->world_to_object, dpdu);v->dOdv=transform_vector(hg->world_to_object, dpdv);
    if (contexts & (1u << 1)) {
      float3 extent=hi-lo;
      v->dGdu=make_float3(extent.x>0?v->dOdu.x/extent.x:0,extent.y>0?v->dOdu.y/extent.y:0,extent.z>0?v->dOdu.z/extent.z:0);
      v->dGdv=make_float3(extent.x>0?v->dOdv.x/extent.x:0,extent.y>0?v->dOdv.y/extent.y:0,extent.z>0?v->dOdv.z/extent.z:0);
    }
  }
  v->Ng = Ng;
  v->Ns = Ns;
  v->T = T;
  v->uv = uv;
  v->vc = vc;
  v->vertex_alpha=(contexts & (1u << 8)) && hg->vertex_alphas?hg->vertex_alphas[i0]*b0+hg->vertex_alphas[i1]*bary.x+hg->vertex_alphas[i2]*bary.y:1;
  // Hash the OptiX instance id into a uniform [0,1) for ObjectInfo.Random.
  // PCG-style multiplicative hash: cheap, decorrelated, deterministic per
  // instance — what Cycles uses for the same node.
  unsigned int iid = hg->instance_id;
  unsigned int h = iid * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  h = (h >> 22u) ^ h;
  v->object_random = (float)h * (1.0f / 4294967296.0f);
  v->mesh_light_idx = hg->area_light_group;
  v->prim_area = prim_area;
  v->hit = 1;
}

static __device__ void fill_path_vertex(PathVertex *v) {
  fill_surface_vertex(v,(HitGroupData*)optixGetSbtDataPointer(),optixGetPrimitiveIndex(),optixGetTriangleBarycentrics(),-optixGetWorldRayDirection(),optixGetRayTmax(),(optixGetRayFlags()&OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT)?2u:1u);
}

// Geometry-only node queries use compact software BVHs. No OptiX trace or
// continuation is entered from the interpreter (including shadow any-hit).
