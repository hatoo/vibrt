// ---------- Path tracing payload ----------
struct PathVertex {
  float3 P;  // hit position
  float3 Ng; // geometric normal (world)
  float3 Ns; // shading normal (world)
  float3 T;  // tangent (world)
  float2 uv; // mesh UVs (0,0 if absent)
  float3 vc; // interpolated vertex colour (1,1,1 if mesh has none)
  PrincipledGpu *mat;
  // Per-instance uniform random in [0..1). Derived from
  // `optixGetInstanceId()` via a hash so each particle / instance gets a
  // stable but-different value — feeds `COLOR_NODE_OBJECT_RANDOM` so the
  // material graph can lerp ColorRamps and Mix nodes per instance.
  float object_random;
  float3 P_object, generated, N_object, object_location, incoming;
  float3 dPdu, dPdv;
  float3 dOdu, dOdv, dGdu, dGdv;
  HitGroupData *geometry;
  unsigned int primitive, shader_seed;
  float2 barycentric;
  float vertex_alpha;
  float ray_length;
  unsigned int path_depth, diffuse_depth, glossy_depth, transmission_depth, path_flags;
  int hit; // 1 if hit, 0 if miss
  // Mesh-light index for the instance this hit landed on (-1 when the
  // instance isn't registered for mesh-light NEE). Mirrors the SBT's
  // `HitGroupData::area_light_group`. Lets `trace_path` decide whether
  // a BSDF-sampled emission contribution should be MIS-weighted against
  // the NEE sampler — without indexing into the mesh_lights array, since
  // the joint pmf reduces to `area × lum / total_power` regardless of
  // which mesh-light owns the triangle.
  int mesh_light_idx;
  // World-space triangle area of the hit primitive. Closest_hit computes
  // it from the post-transform vertices so `trace_path`'s MIS gate can
  // form `pdf_solid = d² / (area × cos_light)` without re-fetching the
  // BLAS triangle. Zero on misses / non-mesh-light hits.
  float prim_area;
  float light_pdf;
};

static __forceinline__ __device__ void *unpack_ptr(unsigned int hi,
                                                   unsigned int lo) {
  return (void *)(((unsigned long long)hi << 32) | (unsigned long long)lo);
}
static __forceinline__ __device__ void pack_ptr(void *p, unsigned int &hi,
                                                unsigned int &lo) {
  unsigned long long u = (unsigned long long)p;
  hi = (unsigned int)(u >> 32);
  lo = (unsigned int)(u & 0xFFFFFFFFu);
}

static __forceinline__ __device__ PathVertex *get_path_vertex() {
  return (PathVertex *)unpack_ptr(optixGetPayload_0(), optixGetPayload_1());
}
