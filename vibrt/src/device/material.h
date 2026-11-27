// ---------- Colour graph evaluator ----------
static __device__ float volume_phase_eval(const Volume *vol, float cosine) {
#if VIBRT_EXTENDED_PHASES
  return vol->phase == 0
             ? phase_hg_eval(vol->anisotropy, cosine)
             : vibrt_shader::volume_phase_pdf(vol->phase, vol->phase_params,
                                              vol->anisotropy, cosine);
#else
  return phase_hg_eval(vol->anisotropy, cosine);
#endif
}
static __device__ float3 volume_phase_sample(const Volume *vol, float u,
                                             float v, float &pdf) {
#if VIBRT_EXTENDED_PHASES
  if (vol->phase == 0)
    return phase_hg_sample(vol->anisotropy, u, v, pdf);
  auto result = vibrt_shader::volume_phase_sample(vol->phase, vol->phase_params,
                                                  vol->anisotropy, u, v, &pdf);
  return make_float3(result.x, result.y, result.z);
#else
  return phase_hg_sample(vol->anisotropy, u, v, pdf);
#endif
}

static __device__ void shader_context(const PathVertex &v, vibrt_cycles::float4 *ctx,
                                      unsigned int mask, int derivative = 0) {
  using vibrt_cycles::make_float4;
  mask &= VIBRT_SHADER_CONTEXTS;
#define NEED(I) (mask & (1u << (I)))
#define CTX_VEC(I,V) if(NEED(I))ctx[I]=make_float4((V).x,(V).y,(V).z,1)
  constexpr float delta = 0.001f;
  float du = derivative == 1 ? v.du.dx * delta : derivative == 2 ? v.du.dy * delta : 0;
  float dv = derivative == 1 ? v.dv.dx * delta : derivative == 2 ? v.dv.dy * delta : 0;
  if(NEED(0))ctx[0]=make_float4(v.uv.x+du,v.uv.y+dv,0,1);
  if(NEED(1)) {
    float3 G=v.generated;
    G=G+v.dGdu*du+v.dGdv*dv;
    CTX_VEC(1,G);
  }
  if(NEED(2)) {
    float3 O=v.P_object;
    O=O+v.dOdu*du+v.dOdv*dv;
    CTX_VEC(2,O);
  }
  float3 P=v.P;
  if(mask&((1u<<3)|(1u<<12)|(1u<<14))) {
    auto dP = vibrt_differential::differential_from_compact(cycles_vector(v.Ng), v.dP);
    auto offset = derivative == 1 ? dP.dx * delta : derivative == 2 ? dP.dy * delta : vibrt_cycles::make_float3(0);
    P = P + make_float3(offset.x, offset.y, offset.z);
    CTX_VEC(3,P);
  }
  CTX_VEC(4,v.Ns);CTX_VEC(5,v.Ng);CTX_VEC(6,v.incoming);
  if(NEED(7)||NEED(17)) {
    float3 T=v.dPdu,B;
    if(dot3(T,T)<1e-8f)build_frame(v.Ns,T,B);
    else T=normalize3(T-v.Ns*dot3(T,v.Ns));
    CTX_VEC(7,T);
    if(NEED(17))ctx[17]=make_float4(dot3(cross3(v.Ns,T),v.dPdv)<0?-1.0f:1.0f,0,0,0);
  }
  if(NEED(8))ctx[8]=make_float4(v.vc.x,v.vc.y,v.vc.z,v.vertex_alpha);
  if(NEED(9))ctx[9]=make_float4(v.object_random,v.geometry?(float)v.geometry->object_index:0,0,v.geometry?v.geometry->object_color[3]:1);
  CTX_VEC(10,v.object_location);
  if(NEED(11))ctx[11]=v.geometry?make_float4(v.geometry->object_color[0],v.geometry->object_color[1],v.geometry->object_color[2],v.geometry->object_color[3]):make_float4(1);
  if(NEED(12)||NEED(14)) {
    float3 cp=P-make_f3(params.cam_eye);
    if(NEED(12)) {
      float3 cr=make_float3(dot3(cp,normalize3(make_f3(params.cam_u))),dot3(cp,normalize3(make_f3(params.cam_v))),dot3(cp,normalize3(make_f3(params.cam_w))));
      CTX_VEC(12,cr);
    }
    if(NEED(14)) {
      float z=params.cam_orthographic ? 1.0f : dot3(cp,normalize3(make_f3(params.cam_w)));
      ctx[14]=make_float4(0.5f-params.cam_shift_x+0.5f*dot3(cp,make_f3(params.cam_u))/fmaxf(z*dot3(make_f3(params.cam_u),make_f3(params.cam_u)),1e-8f),0.5f-params.cam_shift_y+0.5f*dot3(cp,make_f3(params.cam_v))/fmaxf(z*dot3(make_f3(params.cam_v),make_f3(params.cam_v)),1e-8f),0,1);
    }
  }
  CTX_VEC(13,v.N_object);
  if(NEED(15)) {
    float3 refl=2.0f*dot3(v.Ns,v.incoming)*v.Ns-v.incoming;
    CTX_VEC(15,refl);
  }
  if(NEED(16))ctx[16]=vibrt_shader::scalar(dot3(v.Ng,v.incoming)<0?1.0f:0.0f);
  if(NEED(18))ctx[18]=make_float4(v.ray_length,(float)v.path_depth,(float)v.diffuse_depth,(float)v.glossy_depth);
  if(NEED(19))ctx[19]=make_float4((float)v.transmission_depth,0,0,0);
  if(NEED(20))for(int i=0;i<4;i++)ctx[20][i]=(v.path_flags&(1u<<i))?1.0f:0.0f;
  if(NEED(21))for(int i=0;i<4;i++)ctx[21][i]=(v.path_flags&(1u<<(i+4)))?1.0f:0.0f;
  if(mask&(0x3fu<<22))for(int i=0;i<3;i++) {
    if(NEED(22+i)) {
      const float *a=v.geometry?v.geometry->object_to_world+4*i:nullptr;
      ctx[22+i]=a?make_float4(a[0],a[1],a[2],a[3]):make_float4(0);
    }
    if(NEED(25+i)) {
      const float *b=v.geometry?v.geometry->world_to_object+4*i:nullptr;
      ctx[25+i]=b?make_float4(b[0],b[1],b[2],b[3]):make_float4(0);
    }
  }
  if(mask&(7u<<28)) {
    float3 axes[3]={normalize3(make_f3(params.cam_u)),normalize3(make_f3(params.cam_v)),normalize3(make_f3(params.cam_w))};
    for(int i=0;i<3;i++)if(NEED(28+i))ctx[28+i]=make_float4(axes[i].x,axes[i].y,axes[i].z,-dot3(axes[i],make_f3(params.cam_eye)));
  }
  if(NEED(31))ctx[31]=make_float4(0);
#undef CTX_VEC
#undef NEED
}
static __device__ __noinline__ void evaluate_shader_vm_impl(PrincipledGpu *m,const PathVertex &v,vibrt_cycles::float4 *out,unsigned int requested,int derivative) {
  vibrt_cycles::float4 ctx[VIBRT_SHADER_CONTEXT_SIZE];shader_context(v,ctx,m->shader_contexts,derivative);
  vibrt_shader::SceneServices services{&v, derivative};
  vibrt_shader::evaluate_vm((const ShaderInstruction *)m->shader_nodes,
                            m->shader_len, ctx, out, requested, &services);
}
// Keep the tracing interpreter behind a continuation-callable boundary so
// OptiX does not duplicate it into every material/light/shadow evaluation site.
static __device__ void evaluate_shader_vm(PrincipledGpu *m, const PathVertex &v,
                                        vibrt_cycles::float4 *out, unsigned int requested, int derivative) {
#if (VIBRT_SHADER_OPS & (7ull << 41))
  optixContinuationCall<void, PrincipledGpu *, const PathVertex *, vibrt_cycles::float4 *, unsigned int, int>(
      1, m, &v, out, requested, derivative);
#else
  evaluate_shader_vm_impl(m, v, out, requested, derivative);
#endif
}
static __device__ __noinline__ void evaluate_shader_single(PrincipledGpu *m,float2 uv,vibrt_cycles::float4 *out,unsigned int requested) {
  const ShaderInstruction &n=*(const ShaderInstruction*)m->shader_nodes;
  if ((n.op >> 8) & requested)
    vibrt_shader::evaluate_single(n,vibrt_cycles::make_float4(uv.x,uv.y,0,1),out);
}
static __device__ __forceinline__ void evaluate_shader(PrincipledGpu *m,const PathVertex &v,vibrt_cycles::float4 *out,unsigned int requested=vibrt_shader::EVAL_ALL,int derivative=0) {
  if (m->shader_len == 1) {
    float2 uv=v.uv;
    if(derivative==1){uv.x+=v.du.dx*0.001f;uv.y+=v.dv.dx*0.001f;}
    else if(derivative==2){uv.x+=v.du.dy*0.001f;uv.y+=v.dv.dy*0.001f;}
    evaluate_shader_single(m,uv,out,requested);
  } else evaluate_shader_vm(m,v,out,requested,derivative);
}
static __device__ float3 eval_color_graph(
    ColorGraphNode *nodes, int n, int output, float2 uv, float3 vc, float random) {
  auto value = vibrt_nodes::eval_color_graph(nodes, n, output, {uv.x, uv.y}, {vc.x, vc.y, vc.z}, random);
  return make_float3(value.x, value.y, value.z);
}

static __device__ float3 volume_emission(const Volume *v, unsigned int flags) {
  if (!v->emission_nodes) return make_f3(v->emission);
  PrincipledGpu m = {};
  m.shader_nodes = v->emission_nodes;
  m.shader_len = v->emission_len;
  m.shader_contexts = v->emission_contexts;
  PathVertex vertex = {};
  vertex.path_flags = flags;
  vibrt_cycles::float4 out[VIBRT_SHADER_OUTPUT_COUNT];
  evaluate_shader(&m, vertex, out);
  return make_float3(fmaxf(out[5].x,0), fmaxf(out[5].y,0), fmaxf(out[5].z,0));
}

// ---------- Principled BSDF ----------
//
// Material evaluation is a two-step process. `eval_material` walks the
// surface's binary Mix tree (`PrincipledGpu` left/right children) and
// flattens it into a flat list of (closure, weight) pairs — one
// `MaterialEval` per leaf. `eval_bsdf` / `sample_bsdf` then evaluate as
// a weighted sum / mixture over those leaves. This handles arbitrary
// binary topologies (right-leaning `Mix(A, Mix(B, C))`, left-leaning
// `Mix(Mix(A, B), C)`, balanced `Mix(Mix(A, B), Mix(C, D))`) uniformly.
struct MaterialEval {
  float3 base_color;
  float metallic;
  float roughness;
  float alpha;   // isotropic α = roughness², used for LUT + transmission
  float alpha_x; // anisotropic αx (along tangent T)
  float alpha_y; // anisotropic αy (along bitangent B)
  // Filter-glossy-inflated coat α. Pre-computed in eval_material so the
  // clearcoat lobe in eval_bsdf / sample_bsdf doesn't have to re-derive
  // it from the material's coat_roughness on every call (and so it
  // picks up the path's min_alpha state along with the main lobe).
  float coat_alpha;
  float ior;
  float transmission;
  float3 emission;
  float3 Ns; // shading normal
  float3 Ng; // geometric normal (for shadow-terminator term)
  float3 T;  // rotated by tangent_rotation for anisotropy
  float3 B;
  float specular_ior_level, anisotropy, tangent_rotation;
  float coat_weight, coat_roughness, coat_ior;
  float sheen_weight, sheen_roughness, sss_weight, translucent_weight;
  float hair_offset, hair_roughness_u, hair_roughness_v, alpha_blend;
  float3 sheen_tint, sss_radius;
  float3 specular_tint, coat_tint, coat_normal, conductor_ior, conductor_k;
  float thin_film_thickness, thin_film_ior, diffuse_roughness;
#if (VIBRT_CLOSURE_TYPES & (1u << 7))
  float sss_scale, sss_ior, sss_anisotropy;
  bool sss_exit;
#endif
#if VIBRT_CLOSURE_TYPES
  float toon_size, toon_smooth;
  float3 portal_position, portal_direction;
#endif
#if (VIBRT_CLOSURE_TYPES & (1u << 5))
  vibrt_cycles::ChiangHairBSDF chiang;
#endif
#if (VIBRT_CLOSURE_TYPES & (1u << 6))
  vibrt_cycles::HuangHairBSDF huang;
  vibrt_cycles::HuangHairExtra huang_extra;
#endif
#if (VIBRT_CLOSURE_TYPES & ((1u << 5) | (1u << 6)))
  vibrt_cycles::ShaderData hair_context;
#endif
  PrincipledGpu *mat; // raw material ptr for coat/sheen/sss params
};

static __device__ vibrt_surface::Parameters surface_parameters(const MaterialEval &e, float3 incoming) {
  using vibrt_cycles::make_float3;
  auto cv = [](::float3 v) { return make_float3(v.x, v.y, v.z); };
  vibrt_surface::Parameters p{};
  p.N = cv(e.Ns); p.Ng = cv(dot3(e.Ng, incoming) < 0 ? -e.Ng : e.Ng);
  p.T = cv(e.T); p.wi = cv(incoming);
  p.coat_normal = vibrt_cycles::safe_normalize_fallback(cv(e.coat_normal), p.N);
  p.color = vibrt_cycles::max(cv(e.base_color), vibrt_cycles::zero_spectrum());
  p.specular_tint = vibrt_cycles::max(cv(e.specular_tint), vibrt_cycles::zero_spectrum());
  p.coat_tint = cv(e.coat_tint);
  p.sheen_tint = vibrt_cycles::max(cv(e.sheen_tint), vibrt_cycles::zero_spectrum());
  p.conductor_ior = cv(e.conductor_ior); p.conductor_k = cv(e.conductor_k);
  p.alpha_x = e.alpha_x; p.alpha_y = e.alpha_y; p.alpha = e.alpha; p.roughness = e.roughness;
  p.backfacing = dot3(e.Ng, incoming) < 0; p.ior = p.backfacing ? 1 / e.ior : e.ior;
  p.metallic = e.metallic; p.transmission = e.transmission;
  p.specular_level = fmaxf(e.specular_ior_level, 0);
  p.diffuse_roughness = __saturatef(e.diffuse_roughness);
  p.coat = e.coat_weight; p.coat_alpha = e.coat_alpha; p.coat_ior = fmaxf(e.coat_ior, 1);
  p.sheen = e.sheen_weight; p.sheen_roughness = e.sheen_roughness;
  p.thin_film_thickness = e.thin_film_thickness; p.thin_film_ior = fmaxf(e.thin_film_ior, 1e-5f);
  p.subsurface = e.sss_weight; p.translucent = e.translucent_weight;
  p.model = e.mat->surface_model; p.distribution = e.mat->distribution;
  if (!p.model && e.mat->pure_glossy) p.model = vibrt_surface::GLOSSY;
  if (!p.model && e.mat->pure_diffuse)
    p.model = e.sheen_weight > 0 ? vibrt_surface::SHEEN : vibrt_surface::DIFFUSE;
  return p;
}

// Shared only by direct lighting and direction sampling at one surface hit.
// The leaves and outgoing direction stay fixed for that pair of operations.
// Keep the storage in place: Closures contains pointers into its own members.
struct SurfaceClosureCache {
  vibrt_surface::Closures closures;
  const MaterialEval *material;
};

static __forceinline__ __device__ const vibrt_surface::Closures &surface_closures(
    const MaterialEval &e, float3 wo, SurfaceClosureCache &cache) {
  if (cache.material != &e) {
    vibrt_surface::setup(surface_parameters(e, wo), cache.closures);
    cache.material = &e;
  }
  return cache.closures;
}

// Group of leaf MaterialEvals + their per-leaf weights. The weights sum
// to 1 across the group (modulo per-pixel mix_fac quantisation). Built
// by `eval_material` from the binary Mix tree at the hit. The host validates
// the tree and compiles exactly the maximum leaf count used by the scene.
constexpr int MAX_LEAVES = VIBRT_MAX_CLOSURES;
struct MaterialEvalGroup {
  MaterialEval *slots; // n elements
  float *weights;      // n elements, sum to ≈1
  int n;
};

// `min_alpha` is the path's "filter glossy" floor — Cycles' filter-glossy
// settings inflate the BSDF roughness on indirect glossy bounces so that
// near-mirror bounces can't resample a tiny solid angle and produce
// fireflies. We approximate it by raising `alpha`, `alpha_x`, `alpha_y`
// to at least this value. 0 = primary ray (no inflation).
//
// Single-closure eval — takes an explicit `PrincipledGpu* m` so the same
// body can serve both the primary closure (m = v.mat) and the secondary
// (m = v.mat->secondary). The vertex-level inputs (uv, vertex color,
// tangents, normals) come from `v` regardless of which closure is being
// evaluated.
struct ShaderEvalCache {
  unsigned long long nodes;
  vibrt_cycles::float4 outputs[VIBRT_SHADER_OUTPUT_COUNT];
  float3 normal, tangent;
};

static __forceinline__ __device__ unsigned long long shader_outputs(const PrincipledGpu *m) {
  // The host includes surface, closure-child, AOV and volume graphs in this
  // union. Constant bits let NVRTC omit unused material-output branches.
  return m->shader_outputs & VIBRT_SHADER_OUTPUTS;
}

static __device__ __noinline__ MaterialEval eval_material_one(const PathVertex &v,
                                                  PrincipledGpu *m,
                                                  ShaderEvalCache &cache,
                                                  float min_alpha = 0.0f) {
  MaterialEval e;
  e.mat = m;
#if VIBRT_CLOSURE_TYPES
  e.toon_size = 0.5f;
  e.toon_smooth = 0;
  e.portal_position = v.P;
  e.portal_direction = -v.incoming;
#endif
  e.base_color = make_f3(m->base_color);
  e.specular_tint = make_f3(m->specular_tint);
  e.coat_tint = make_f3(m->coat_tint);
  e.coat_normal = make_f3(m->coat_normal);
  e.thin_film_thickness = m->thin_film_thickness;
  e.thin_film_ior = m->thin_film_ior;
  e.conductor_ior = make_f3(m->conductor_ior);
  e.conductor_k = make_f3(m->conductor_k);
  e.diffuse_roughness = m->diffuse_roughness;
  e.metallic = m->metallic;
  e.roughness = m->roughness;
  e.ior = m->ior;
  e.transmission = m->transmission;
  e.emission = make_f3(m->emission);
  e.specular_ior_level=m->specular_ior_level;e.anisotropy=m->anisotropy;e.tangent_rotation=m->tangent_rotation;
  e.coat_weight=m->coat_weight;e.coat_roughness=m->coat_roughness;e.coat_ior=m->coat_ior;
  e.sheen_weight=m->sheen_weight;e.sheen_roughness=m->sheen_roughness;e.sheen_tint=make_f3(m->sheen_tint);
  e.sss_weight=m->sss_weight;e.sss_radius=make_f3(m->sss_radius);e.translucent_weight=m->translucent_weight;
#if (VIBRT_CLOSURE_TYPES & (1u << 7))
  e.sss_scale=m->sss_scale;e.sss_ior=m->sss_ior;e.sss_anisotropy=m->sss_anisotropy;
  e.sss_exit=false;
#endif
  e.hair_offset=m->hair_offset;e.hair_roughness_u=m->hair_roughness_u;e.hair_roughness_v=m->hair_roughness_v;e.alpha_blend=m->alpha_blend;

  const float *M = m->uv_transform;
  float2 uv =
      make_float2(M[0] * v.uv.x + M[1] * v.uv.y + M[2],
                  M[3] * v.uv.x + M[4] * v.uv.y + M[5]);

  if (m->color_graph_nodes != nullptr && m->color_graph_len > 0) {
    // Graph evaluation reads the raw mesh UV; each ImageTex node carries
    // its own uv_transform so sources with different Mapping nodes don't
    // get jammed into one global scale.
    float3 g = eval_color_graph(m->color_graph_nodes, m->color_graph_len,
                                m->color_graph_output, v.uv, v.vc,
                                v.object_random);
    e.base_color = e.base_color * g;
  } else if (m->base_color_tex != nullptr) {
    float3 t =
        sample_rgba(m->base_color_tex, m->base_color_tex_w, m->base_color_tex_h,
                    m->base_color_tex_channels, uv);
    e.base_color = e.base_color * t;
  }
  if (m->use_vertex_color) {
    e.base_color = e.base_color * v.vc;
  }
  if (m->roughness_tex != nullptr) {
    float3 t = sample_rgba(m->roughness_tex, m->roughness_tex_w,
                           m->roughness_tex_h, m->roughness_tex_channels, uv);
    e.roughness = e.roughness * t.x;
  }
  if (m->metallic_tex != nullptr) {
    float3 t = sample_rgba(m->metallic_tex, m->metallic_tex_w,
                           m->metallic_tex_h, m->metallic_tex_channels, uv);
    e.metallic = e.metallic * t.x;
  }
  if (m->transmission_tex != nullptr) {
    float3 t =
        sample_rgba(m->transmission_tex, m->transmission_tex_w,
                    m->transmission_tex_h, m->transmission_tex_channels, uv);
    e.transmission = e.transmission * t.x;
  }
  if (m->emission_tex != nullptr) {
    float3 t =
        sample_rgba(m->emission_tex, m->emission_tex_w,
                    m->emission_tex_h, m->emission_tex_channels, uv);
    e.emission = e.emission * t;
  }
  e.metallic = fminf(fmaxf(e.metallic, 0.0f), 1.0f);
  float3 graph_normal=v.Ns,graph_tangent=v.T;
  if(m->shader_nodes && m->shader_len>0) {
    vibrt_cycles::float4 *out=cache.outputs;
    bool cached=cache.nodes==m->shader_nodes;
    if(!cached)evaluate_shader(m,v,out);
    unsigned long long mask = shader_outputs(m);
#define GRAPH_SCALAR(I, F)                                                     \
  if (mask & (1ull << I))                                                      \
  e.F = out[I].x
#define GRAPH_COLOR(I, F)                                                      \
  if (mask & (1ull << I))                                                      \
  e.F = make_float3(out[I].x, out[I].y, out[I].z)
    GRAPH_COLOR(0,base_color);GRAPH_SCALAR(1,roughness);GRAPH_SCALAR(2,metallic);GRAPH_SCALAR(3,ior);
    GRAPH_SCALAR(4,transmission);GRAPH_COLOR(5,emission);if(mask&(1u<<6))e.emission=e.emission*out[6].x;
    GRAPH_SCALAR(7,alpha_blend);GRAPH_SCALAR(8,specular_ior_level);GRAPH_SCALAR(9,anisotropy);
    if(mask&(1u<<10))e.tangent_rotation=out[10].x*6.283185307179586f;
    GRAPH_SCALAR(11,coat_weight);GRAPH_SCALAR(12,coat_roughness);GRAPH_SCALAR(13,coat_ior);
    GRAPH_SCALAR(14,sheen_weight);GRAPH_SCALAR(15,sheen_roughness);GRAPH_COLOR(16,sheen_tint);
    GRAPH_SCALAR(17,sss_weight);GRAPH_COLOR(18,sss_radius);GRAPH_SCALAR(19,translucent_weight);
    GRAPH_SCALAR(20,hair_offset);GRAPH_SCALAR(21,hair_roughness_u);GRAPH_SCALAR(22,hair_roughness_v);
    GRAPH_COLOR(31, conductor_ior);
#if VIBRT_SHADER_OUTPUT_COUNT > 32
    GRAPH_COLOR(36, specular_tint); GRAPH_COLOR(37, coat_tint); GRAPH_COLOR(38, coat_normal);
    GRAPH_SCALAR(39, thin_film_thickness); GRAPH_SCALAR(40, thin_film_ior);
    GRAPH_COLOR(48, conductor_k);
    GRAPH_SCALAR(49, diffuse_roughness);
#endif
#if (VIBRT_CLOSURE_TYPES & (1u << 7)) && VIBRT_SHADER_OUTPUT_COUNT > 32
    GRAPH_SCALAR(41, sss_scale);
    GRAPH_SCALAR(42, sss_ior);
    GRAPH_SCALAR(43, sss_anisotropy);
#endif
#if (VIBRT_CLOSURE_TYPES & ((1u << 5) | (1u << 6)))
    if (m->closure_type == 5 || m->closure_type == 6) {
      e.hair_context = {};
      e.hair_context.wi =
          vibrt_cycles::make_float3(v.incoming.x, v.incoming.y, v.incoming.z);
      float3 tangent = dot3(v.T, v.T) > 1e-10f ? v.T : v.dPdu;
      e.hair_context.dPdu =
          vibrt_cycles::make_float3(tangent.x, tangent.y, tangent.z);
      e.hair_context.Ng = vibrt_cycles::make_float3(v.Ng.x, v.Ng.y, v.Ng.z);
      e.hair_context.N = vibrt_cycles::make_float3(v.Ns.x, v.Ns.y, v.Ns.z);
      e.hair_context.lcg_state =
          vibrt_cycles::hash_uint2(v.shader_seed, 0x68bc21ebu);
      e.hair_context.type = v.geometry->curve_radii ? 1 :
          (v.geometry->tangents ? vibrt_cycles::PRIMITIVE_CURVE_RIBBON : 0);
      e.hair_context.v = v.uv.x * 2 - 1;
#if (VIBRT_CLOSURE_TYPES & (1u << 5))
      if (m->closure_type == 5)
      e.chiang = vibrt_shader::chiang_setup(
          e.hair_context,
          vibrt_cycles::make_float3(e.base_color.x, e.base_color.y,
                                    e.base_color.z),
          vibrt_shader::xyz(out[44]), e.hair_roughness_u, e.hair_roughness_v,
          e.hair_offset, out[47].x, out[46].x, out[45].x, out[52].x, out[53].x,
          out[54].x, out[55].x, vibrt_shader::xyz(out[56]), (int)out[57].x);
#endif
#if (VIBRT_CLOSURE_TYPES & (1u << 6))
      if (m->closure_type == 6) {
        e.huang = {}; e.huang_extra = {};
        e.huang.extra = &e.huang_extra;
        e.huang.weight = vibrt_cycles::one_spectrum();
        e.huang.roughness = e.hair_roughness_u * (1 + 2 * (out[45].x - 0.5f) * out[52].x);
        e.huang.sigma = vibrt_shader::hair_sigma(cycles_vector(e.base_color),
            vibrt_shader::xyz(out[44]), e.huang.roughness, out[45].x, out[53].x,
            out[54].x, out[55].x, vibrt_shader::xyz(out[56]), (int)out[57].x);
        e.huang.eta = fmaxf(out[47].x, 1.0001f);
        e.huang.tilt = e.hair_offset;
        e.huang.aspect_ratio = fmaxf(out[46].x, 1e-4f);
        e.huang_extra.R = fmaxf(e.hair_roughness_v, 0);
        e.huang_extra.TT = fmaxf(out[50].x, 0);
        e.huang_extra.TRT = fmaxf(out[51].x, 0);
        e.huang_extra.pixel_coverage = (v.path_flags & 1u) && v.path_depth == 0 && v.curve_radius > 0 ?
            fmaxf(0.5f * v.dP / v.curve_radius, 1e-5f) : 1.0f;
        if (v.geometry->curve_radii && v.geometry->normals) {
          const int *ids = v.geometry->indices + 3 * v.primitive;
          float3 normal = make_f3(v.geometry->normals + 3 * ids[0]) * (1 - v.barycentric.x) +
                          make_f3(v.geometry->normals + 3 * ids[1]) * v.barycentric.x;
          e.huang.N = cycles_vector(transform_normal(v.geometry->world_to_object, normal));
        }
        vibrt_cycles::bsdf_hair_huang_setup(&e.hair_context, &e.huang, 0);
        if (e.hair_context.transparent) e.alpha_blend = 0;
      }
#endif
    }
#endif
#if VIBRT_CLOSURE_TYPES && VIBRT_SHADER_OUTPUT_COUNT > 32
    if (mask & (1ull << 32))
      e.toon_size = out[32].x;
    if (mask & (1ull << 33))
      e.toon_smooth = out[33].x;
    if (mask & (1ull << 34))
      e.portal_position = make_float3(out[34].x, out[34].y, out[34].z);
    if (mask & (1ull << 35)) {
      float3 d = make_float3(out[35].x, out[35].y, out[35].z);
      if (dot3(d, d) > 1e-20f)
        e.portal_direction = normalize3(d);
    }
#endif
    if(mask&(1u<<24))graph_normal=normalize3(make_float3(out[24].x,out[24].y,out[24].z));
    if(mask&(1u<<25))graph_tangent=normalize3(make_float3(out[25].x,out[25].y,out[25].z));
    if(cached) {
      graph_normal=cache.normal;graph_tangent=cache.tangent;
    } else if(mask&(1u<<26)) {
      constexpr float delta=0.001f;
      vibrt_cycles::float4 shifted[VIBRT_SHADER_OUTPUT_COUNT];
      evaluate_shader(m,v,shifted,vibrt_shader::EVAL_BUMP,1);
      float dhdu=(shifted[26].x-out[26].x)/delta;
      evaluate_shader(m,v,shifted,vibrt_shader::EVAL_BUMP,2);
      float dhdv=(shifted[26].x-out[26].x)/delta;
      vibrt_scene_nodes::ShaderData sd{};
      sd.N = cycles_vector(graph_normal);
      auto dP = vibrt_differential::differential_from_compact(cycles_vector(v.Ng), v.dP);
      auto bumped = vibrt_scene_nodes::bump(sd, dP, 0, dhdu, dhdv, out[27].x, out[28].x, 1, false, false);
      graph_normal = make_float3(bumped.x, bumped.y, bumped.z);
    }
    cache.nodes=m->shader_nodes;cache.normal=graph_normal;cache.tangent=graph_tangent;
#undef GRAPH_SCALAR
#undef GRAPH_COLOR
  }
  e.alpha_blend=fminf(fmaxf(e.alpha_blend,0.0f),1.0f);
  e.sss_weight=m->sss_method ? __saturatef(e.sss_weight) : 0.0f;
  e.emission=e.emission*e.alpha_blend;
  e.metallic=fminf(fmaxf(e.metallic,0.0f),1.0f);
  e.roughness=fminf(fmaxf(e.roughness,0.0f),1.0f);
  e.ior=fmaxf(e.ior,1e-5f);
  e.transmission = fminf(fmaxf(e.transmission, 0.0f), 1.0f);
  // No separate roughness floor — the alpha clamp below is the sampler's
  // numerical safeguard; forcing a minimum roughness on top of it would
  // silently turn artist-authored mirrors into slightly-rough dielectrics.
  e.alpha = e.roughness * e.roughness;
  // Filter-glossy: the path tracer raises `min_alpha` once it's already
  // taken at least one glossy / specular bounce so that subsequent
  // near-mirror evals can't refocus the radiance into a tiny solid
  // angle. Approximates Cycles' Light Paths > Filter Glossy. min_alpha=0
  // (primary ray) leaves alpha untouched.
  if (min_alpha > e.alpha)
    e.alpha = min_alpha;

  float3 Ns = graph_normal;
  if (m->bump_tex || m->normal_tex) {
    vibrt_scene_nodes::ShaderData sd{};
    sd.N = sd.normal = cycles_vector(Ns);
    sd.Ng = cycles_vector(v.Ng);
    sd.sign = 1;
    float3 T, B;
    build_frame(Ns, T, B);
    sd.tangent = cycles_vector(T);
    if (m->bump_tex && m->bump_strength != 0) {
      constexpr float filter = 0.1f;
      auto height = [&](float dx, float dy) {
        float du = v.du.dx * dx + v.du.dy * dy;
        float dv = v.dv.dx * dx + v.dv.dy * dy;
        return sample_rgba(m->bump_tex, m->bump_tex_w, m->bump_tex_h, m->bump_tex_channels,
                           make_float2(uv.x + M[0]*du + M[1]*dv, uv.y + M[3]*du + M[4]*dv)).x;
      };
      auto dP = vibrt_differential::differential_from_compact(sd.Ng, v.dP);
      sd.N = sd.normal = vibrt_scene_nodes::bump(sd, dP, height(0,0), height(filter,0),
          height(0,filter), 1, m->bump_strength, filter, false, false);
    }
    if (m->normal_tex) {
      float3 value = sample_rgba(m->normal_tex, m->normal_tex_w, m->normal_tex_h,
                                m->normal_tex_channels, uv);
      sd.N = vibrt_scene_nodes::normal_map(sd, cycles_vector(value), m->normal_strength,
                                          0, false, false);
    }
    Ns = make_float3(sd.N.x, sd.N.y, sd.N.z);
  }
  e.Ns = Ns;
  e.Ng = v.Ng;
  if (m->hair_weight > 0.5f && !(m->shader_outputs & (1ull << 25)) && dot3(v.T, v.T) < 1e-10f) {
    // Cycles uses dPdv and no offset for a mesh with an unlinked Hair tangent.
    graph_tangent = v.dPdv;
    e.hair_offset = 0;
  }
  // If closesthit interpolated an authored tangent (hair strand axis), use
  // it as the anisotropy / hair tangent after re-orthogonalising against the
  // possibly-perturbed shading normal. Otherwise fall back to a synthetic
  // tangent — the previous behaviour for every non-hair surface.
  if (dot3(graph_tangent, graph_tangent) > 0.5f) {
    float3 Tperp = graph_tangent - e.Ns * dot3(graph_tangent, e.Ns);
    float Tn2 = dot3(Tperp, Tperp);
    if (Tn2 > 1e-8f) {
      e.T = Tperp * (1.0f / sqrtf(Tn2));
      e.B = normalize3(cross3(e.Ns, e.T));
    } else {
      build_frame(e.Ns, e.T, e.B);
    }
  } else {
    build_frame(e.Ns, e.T, e.B);
  }

  if (m->hair_weight > 0.5f && dot3(graph_tangent, graph_tangent) > 1e-12f)
    e.T = normalize3(graph_tangent);

  // Cycles uses a distinct roughness remapping for the standalone Glossy node.
  float aniso = fminf(fmaxf(e.anisotropy, -0.99f), 0.99f);
  if (m->surface_model == vibrt_surface::GLOSSY || (!m->surface_model && m->pure_glossy)) {
    float aspect = 1 - fabsf(aniso);
    e.alpha_x = aniso < 0 ? e.alpha / aspect : e.alpha * aspect;
    e.alpha_y = aniso < 0 ? e.alpha * aspect : e.alpha / aspect;
  } else {
    float aspect = sqrtf(1 - 0.9f * fmaxf(aniso, 0));
    e.alpha_x = e.alpha / aspect;
    e.alpha_y = e.alpha * aspect;
  }
  e.alpha_x = fmaxf(e.alpha_x, min_alpha);
  e.alpha_y = fmaxf(e.alpha_y, min_alpha);
  // Coat α also gets the filter-glossy floor applied. Pebble materials
  // in pabellon use a Fresnel-driven MixShader → my exporter routes the
  // glossy side onto the coat lobe with a low coat_roughness (~0.45),
  // which without inflation can refocus indirect bounces through the
  // pool's surface into a sharper cone than the path has already
  // sampled. Same min_alpha as the main lobe — Cycles' filter_glossy
  // is a single state shared across all glossy lobes.
  e.coat_alpha = e.coat_roughness * e.coat_roughness;
  if (min_alpha > e.coat_alpha)
    e.coat_alpha = min_alpha;
  // Tangent rotation around Ns.
  float rot = e.tangent_rotation;
  if (rot != 0.0f) {
    float c = cosf(rot);
    float s = sinf(rot);
    float3 Trot = e.T * c + e.B * s;
    float3 Brot = e.B * c - e.T * s;
    e.T = normalize3(Trot);
    e.B = normalize3(Brot);
  }
  if (!m->surface_model && !m->pure_diffuse && !m->pure_glossy &&
      (e.coat_weight > 0 || e.sheen_weight > 0) && dot3(e.emission, e.emission) > 0) {
    auto p = surface_parameters(e, v.incoming);
    if (p.backfacing) { p.N = -p.N; p.coat_normal = -p.coat_normal; }
    auto attenuation = vibrt_surface::emission_weight(p);
    e.emission = e.emission * make_float3(attenuation.x, attenuation.y, attenuation.z);
  }
  return e;
}

// Compute the per-pixel mix factor for a single MixShader level, evaluating
// the level's `mix_fac_tex` (if any) at the hit's UV through the level's own
// uv_transform. Cycles' MixShader factor is sampled at the surface point,
// not in either lobe's texture coordinates. Constant-fac is the common case
// (mix_fac_tex == nullptr); the texture branch supports per-pixel masks for
// things like edge-wear blends or multi-material decals.
static __device__ float compute_level_mix_fac(PrincipledGpu *m, const PathVertex &v,
                                              float NoV) {
  if (shader_outputs(m) & (1u << 23)) {
    vibrt_cycles::float4 out[VIBRT_SHADER_OUTPUT_COUNT];
    evaluate_shader(m, v, out, vibrt_shader::EVAL_MIX);
    return fminf(fmaxf(out[23].x, 0.0f), 1.0f);
  }
  float2 uv_raw=v.uv;
  // View-dependent Fresnel mix factor: the exporter encodes a Fresnel /
  // Layer-Weight MixShader factor as a NEGATIVE mix_fac whose magnitude is
  // the Fresnel IOR. Evaluate F(NoV, ior) at the hit so the secondary
  // (right) side gets its true view-angle weight — Blender's clearcoat /
  // "Fresnel over base" idiom (bmw27 CarShellNew: red base at normal, white
  // clearcoat at grazing). A constant hemispheric-mean fac under-weighted
  // the clearcoat on the curved car body (seen mostly at grazing).
  if (m->mix_fac < 0.0f) {
    float ior = -m->mix_fac;
    return fminf(fmaxf(vibrt_cycles::fresnel_dielectric_cos(fabsf(NoV), ior), 0.0f), 1.0f);
  }
  const float *M = m->uv_transform;
  float2 uv_xform =
      make_float2(M[0] * uv_raw.x + M[1] * uv_raw.y + M[2],
                  M[3] * uv_raw.x + M[4] * uv_raw.y + M[5]);
  float f = m->mix_fac;
  if (m->mix_fac_tex != nullptr) {
    float t = sample_rgba(m->mix_fac_tex, m->mix_fac_tex_w,
                          m->mix_fac_tex_h,
                          m->mix_fac_tex_channels, uv_xform).x;
    f *= t;
  }
  return fminf(fmaxf(f, 0.0f), 1.0f);
}

// Walk the material's binary Mix tree and flatten it into the caller-
// provided `slots[]` + `weights[]` arrays as a flat list of leaves.
// Returns the number of leaves emitted (1..max_leaves). Each
// PrincipledGpu node is interpreted as:
//   - secondary == null && left_subtree == null:  leaf (own lobe params)
//   - secondary != null && left_subtree == null:  Mix(self_as_leaf, secondary)
//   - secondary != null && left_subtree != null:  Mix(left_subtree, secondary)
//                                                  (own params unused)
//   - secondary == null && left_subtree != null:  not produced by exporter,
//                                                  treated as leaf for safety
//
// Walks iteratively via an explicit stack — CUDA call recursion is
// supported but slow. Stack depth bounded by `STACK_DEPTH`; with
// at most 16 leaves and 8 levels; STACK_DEPTH=32 covers those validated trees.
//
// `max_leaves == 1` makes this behave like the AOV-pass single-closure
// query (just emit the leftmost leaf). `slots == nullptr` is treated
// the same way (return 0 leaves — caller signalled "I don't care").
// Traced shader graphs otherwise force the entire material evaluator to inline.
struct MaterialCall {
  const PathVertex *vertex;
  MaterialEval *slots;
  float *weights;
  int max_leaves;
  float min_alpha, NoV;
  int count;
};
static __device__ __noinline__ int eval_material_impl(const PathVertex &, MaterialEval *, float *, int, float, float);
static __device__ int eval_material(const PathVertex &v, MaterialEval *slots, float *weights,
                                    int max_leaves, float min_alpha = 0, float NoV = 1) {
#if (VIBRT_SHADER_OPS & (7ull << 41))
  MaterialCall call = {&v, slots, weights, max_leaves, min_alpha, NoV, 0};
  optixContinuationCall<void, MaterialCall *>(2, &call);
  return call.count;
#else
  return eval_material_impl(v, slots, weights, max_leaves, min_alpha, NoV);
#endif
}
static __device__ __noinline__ int eval_material_impl(const PathVertex &v,
                                     MaterialEval *slots,
                                     float *weights,
                                     int max_leaves,
                                     float min_alpha = 0.0f,
                                     float NoV = 1.0f) {
  if (slots == nullptr || max_leaves <= 0) {
    return 0;
  }
  constexpr int STACK_DEPTH = 32;
  struct Frame {
    PrincipledGpu *node;
    float weight;
    bool as_leaf; // true => use own params as a leaf, ignore secondary/left
  };
  Frame stack[STACK_DEPTH];
  int sp = 0;
  stack[sp++] = {v.mat, 1.0f, false};
  // The cache is reset for every intersection. Adjacent closures can share a
  // graph, but values must never leak between objects or path vertices.
  ShaderEvalCache cache;cache.nodes=0;
  int n = 0;
  while (sp > 0 && n < max_leaves) {
    Frame f = stack[--sp];
    if (f.weight == 0.0f)
      continue;
    PrincipledGpu *node = f.node;
    bool has_right = (node->secondary != nullptr);
    bool has_left = (node->left_subtree != nullptr);
    bool is_internal = !f.as_leaf && (has_right || has_left);
    if (!is_internal) {
      // Emit as leaf — evaluate this node's own lobe params.
      slots[n] = eval_material_one(v, node, cache, min_alpha);
      weights[n] = f.weight;
      ++n;
      continue;
    }
    float fac = compute_level_mix_fac(node, v, NoV);
    float w_left = f.weight * (node->add_shader ? 1.0f : 1.0f-fac);
    float w_right = f.weight * (node->add_shader ? 1.0f : fac);
    // Push right then left so left is processed first (preserves
    // depth-first order, helps determinism if anything else inspects
    // the leaf order).
    if (sp + 2 > STACK_DEPTH) {
      // Overflow safeguard: collapse remaining sub-tree into one leaf
      // using own params, weighted by the unspent weight. Should not
      // trigger with a sane exporter cap.
      slots[n] = eval_material_one(v, node, cache, min_alpha);
      weights[n] = f.weight;
      ++n;
      continue;
    }
    if (has_right) {
      stack[sp++] = {node->secondary, w_right, false};
    } else {
      // No right child but left_subtree set: treat node itself as
      // self-leaf carrying right's intended weight (defensive fallback;
      // the exporter doesn't produce this shape).
      stack[sp++] = {node, w_right, true};
    }
    if (has_left) {
      stack[sp++] = {node->left_subtree, w_left, false};
    } else {
      // Right-leaning compact case: own lobe params are the left leaf.
      stack[sp++] = {node, w_left, true};
    }
  }
  return n;
}

// Evaluate BSDF * |cos(wi, Ns)| for a given outgoing wo and incoming wi (both
// world space, pointing away from P). Returns f_r and the sampling PDF used
// by the combined sample routine (for MIS weighting).
struct BsdfEval {
  float3 f;
  float pdf;
};

static __device__ vibrt_cycles::HairBsdf classic_hair(const MaterialEval &e) {
  vibrt_cycles::HairBsdf hair = {};
  hair.N = cycles_vector(e.Ns);
  hair.T = cycles_vector(e.T);
  hair.roughness1 = e.hair_roughness_u;
  hair.roughness2 = e.hair_roughness_v;
  hair.offset = -e.hair_offset;
  if (e.mat->hair_component)
    vibrt_cycles::bsdf_hair_transmission_setup(&hair);
  else
    vibrt_cycles::bsdf_hair_reflection_setup(&hair);
  return hair;
}
#if VIBRT_CLOSURE_TYPES
static __device__ vibrt_cycles::ToonBsdf toon_closure(const MaterialEval &e) {
  vibrt_cycles::ToonBsdf toon = {};
  toon.N = cycles_vector(e.Ns);
  toon.size = e.toon_size;
  toon.smooth = e.toon_smooth;
  if (e.mat->closure_type == 1)
    vibrt_cycles::bsdf_diffuse_toon_setup(&toon);
  else
    vibrt_cycles::bsdf_glossy_toon_setup(&toon);
  return toon;
}
#endif
static __device__ BsdfEval eval_hair(const MaterialEval &e, float3 wo, float3 wi) {
  BsdfEval r;
  auto hair = classic_hair(e);
  auto *sc = reinterpret_cast<const vibrt_cycles::ShaderClosure *>(&hair);
  auto value = e.mat->hair_component
                   ? vibrt_cycles::bsdf_hair_transmission_eval(sc, cycles_vector(wo),
                                                               cycles_vector(wi), &r.pdf)
                   : vibrt_cycles::bsdf_hair_reflection_eval(sc, cycles_vector(wo),
                                                             cycles_vector(wi), &r.pdf);
  r.f = make_float3(value.x, value.y, value.z) * e.base_color * e.alpha_blend;
  r.pdf *= e.alpha_blend;
  return r;
}

// Shadow-terminator softening (Estevez et al. "Hacking the Shadow
// Terminator", matches Cycles' `bump_shadowing_term`). On smooth-shaded
// geometry the interpolated shading normal `N` tilts the reflection
// hemisphere away from the true geometric normal `Ng`, so a reflection
// direction `I` that is above the shading horizon can be *below* the
// geometric horizon — physically self-occluded by the surface. Without
// this term those directions carry full BRDF energy, leaking indirect
// light that compounds per bounce in furniture-dense scenes (classroom /
// junk_shop / lone_monk read ~14% over per diffuse bounce). Returns 1 on
// flat surfaces (Ng == N) so convex / flat-shaded geometry is unaffected.
static __forceinline__ __device__ float bump_shadowing_term(float3 Ng,
                                                            float3 N,
                                                            float3 I) {
  float denom = dot3(N, I) * dot3(Ng, N);
  float g = (denom != 0.0f) ? dot3(Ng, I) / denom : 0.0f;
  g = __saturatef(g); // clamp to [0,1]; endpoints are fixed points of the ramp
  return g * (g * (1.0f - g) + 1.0f); // == -g³ + g² + g (smooth cubic ramp)
}

// Single-closure eval. The legacy body — handles diffuse / spec / coat /
// transmission / sheen / sss / translucent for one PrincipledGpu. The
// outer `eval_bsdf` wrapper below combines this with the secondary
// closure (when present).
static __device__ __noinline__ BsdfEval eval_bsdf_one(const MaterialEval &e, float3 wo,
                                         float3 wi, SurfaceClosureCache &cache) {
  BsdfEval r;
  r.f = make_float3(0, 0, 0);
  r.pdf = 0.0f;
#if (VIBRT_CLOSURE_TYPES & (1u << 7))
  if (e.sss_exit) {
    r.pdf = fmaxf(dot3(e.Ns, wi), 0) * INV_PIf;
    r.f = e.base_color * r.pdf;
    return r;
  }
#endif
#if (VIBRT_CLOSURE_TYPES & (1u << 5))
  if (e.mat->closure_type == 5) {
    auto I = vibrt_cycles::make_float3(wi.x, wi.y, wi.z);
    auto &sd = e.hair_context;
    if (!sd.type && vibrt_cycles::dot(sd.N, I) *
                            vibrt_cycles::dot(sd.N, e.chiang.N) *
                            vibrt_cycles::dot(e.chiang.N, I) <
                        0)
      return r;
    auto value = vibrt_cycles::bsdf_hair_chiang_eval(
        0, &e.hair_context,
        reinterpret_cast<const vibrt_cycles::ShaderClosure *>(&e.chiang),
        vibrt_cycles::make_float3(wi.x, wi.y, wi.z), &r.pdf);
    r.f = make_float3(value.x, value.y, value.z);
    return r;
  }
#endif

#if (VIBRT_CLOSURE_TYPES & (1u << 6))
  if (e.mat->closure_type == 6) {
    if (e.hair_context.transparent) return r;
    auto sd = e.hair_context;
    auto hair = e.huang;
    auto extra = e.huang_extra;
    hair.extra = &extra;
    auto value = vibrt_cycles::bsdf_hair_huang_eval(0, &sd,
        reinterpret_cast<const vibrt_cycles::ShaderClosure *>(&hair), cycles_vector(wi), &r.pdf);
    r.f = make_float3(value.x, value.y, value.z);
    return r;
  }
#endif
  float NoV = dot3(e.Ns, wo);
  if (NoV <= 0.0f)
    return r;
#if VIBRT_CLOSURE_TYPES
  if (e.mat->closure_type) {
    if (e.mat->closure_type <= 2) {
      auto toon = toon_closure(e);
      auto *sc = reinterpret_cast<const vibrt_cycles::ShaderClosure *>(&toon);
      auto value = e.mat->closure_type == 1
                       ? vibrt_cycles::bsdf_diffuse_toon_eval(sc, cycles_vector(wo),
                                                              cycles_vector(wi), &r.pdf)
                       : vibrt_cycles::bsdf_glossy_toon_eval(sc, cycles_vector(wo),
                                                             cycles_vector(wi), &r.pdf);
      r.f = e.base_color * make_float3(value.x, value.y, value.z);
    }
    return r;
  }
#endif
  if (e.mat->hair_weight > 0.5f)
    return eval_hair(e, wo, wi);

  const auto &closures = surface_closures(e, wo, cache);
  auto value = vibrt_surface::evaluate(closures, vibrt_cycles::make_float3(wi.x, wi.y, wi.z), &r.pdf);
  r.f = make_float3(value.x, value.y, value.z);
  if (dot3(e.Ns, wi) > 0) r.f = r.f * bump_shadowing_term(e.Ng, e.Ns, wi);
  r.f = r.f * e.alpha_blend; r.pdf *= e.alpha_blend;
  return r;
}

// Sample BSDF, returning new direction wi, evaluated f*cosθ, and PDF.
// Path-tracer per-type bounce counters key off this lobe id. Cycles caps
// `diffuse_bounces` / `glossy_bounces` / `transmission_bounces` separately
// from the total `max_bounces`, so the kernel needs to know which lobe
// each sampled direction came from.
#define LOBE_DIFFUSE 0
#define LOBE_GLOSSY 1
#define LOBE_TRANSMISSION 2

struct BsdfSample {
#if (VIBRT_CLOSURE_TYPES & (1u << 4))
  bool portal;
  float3 portal_position;
#endif
  float3 wi;
  float3 f;
  float pdf;
  bool specular; // true for rough-specular that shouldn't NEE-double-count
  bool transparent;
  int lobe;      // LOBE_DIFFUSE / LOBE_GLOSSY / LOBE_TRANSMISSION
};

struct BsdfDirectionSample {
  float3 sampled_value;
  float sampled_pdf;
  bool weighted_sample;
  float3 wi;
  bool valid;
  bool specular;
  bool transparent;
  int lobe;
};

// Keep direction sampling separate from evaluation so the large sampler's
// call frame is no longer live while eval_bsdf_one runs.
static __device__ __noinline__ BsdfDirectionSample sample_bsdf_direction(const MaterialEval &e, float3 wo,
                                              RNG &rng, SurfaceClosureCache &cache) {
  BsdfDirectionSample s;
  s.wi = make_float3(0, 0, 1);
  s.valid = false;
  s.specular = false;
  s.lobe = LOBE_DIFFUSE;
  s.transparent = false;
  s.weighted_sample = false;
  if (e.mat->closure_type == 8) {
    s.wi = -wo;
    s.transparent = s.valid = s.specular = true;
    s.lobe = LOBE_TRANSMISSION;
    return s;
  }
#if (VIBRT_CLOSURE_TYPES & (1u << 7))
  if (e.sss_exit) {
    float r = sqrtf(rng.next()), phi = 2 * M_PIf * rng.next();
    s.wi = normalize3(e.T * (r * cosf(phi)) + e.B * (r * sinf(phi)) +
                      e.Ns * sqrtf(fmaxf(1 - r * r, 0)));
    s.valid = true;
    return s;
  }
#endif

#if (VIBRT_CLOSURE_TYPES & (1u << 5))
  if (e.mat->closure_type == 5) {
    auto sd = e.hair_context;
    vibrt_cycles::float3 value, wi;
    vibrt_cycles::float2 roughness;
    float pdf;
    int label = vibrt_cycles::bsdf_hair_chiang_sample(
        0, reinterpret_cast<const vibrt_cycles::ShaderClosure *>(&e.chiang),
        &sd, vibrt_cycles::make_float3(rng.next(), rng.next(), rng.next()),
        &value, &wi, &pdf, &roughness);
    s.wi = make_float3(wi.x, wi.y, wi.z);
    s.valid = pdf > 0;
    s.lobe = (label & vibrt_cycles::LABEL_TRANSMIT) ? LOBE_TRANSMISSION
                                                    : LOBE_GLOSSY;
    s.sampled_value = make_float3(value.x, value.y, value.z);
    s.sampled_pdf = pdf;
    s.weighted_sample = true;
    return s;
  }
#endif

#if (VIBRT_CLOSURE_TYPES & (1u << 6))
  if (e.mat->closure_type == 6) {
    if (e.hair_context.transparent) {
      s.wi = -wo; s.valid = s.transparent = s.specular = true;
      s.lobe = LOBE_TRANSMISSION;
      return s;
    }
    auto sd = e.hair_context;
    auto hair = e.huang;
    auto extra = e.huang_extra;
    hair.extra = &extra;
    vibrt_cycles::float3 value = {}, wi = {};
    vibrt_cycles::float2 roughness;
    float pdf = 0;
    vibrt_cycles::bsdf_hair_huang_sample(0,
        reinterpret_cast<const vibrt_cycles::ShaderClosure *>(&hair), &sd,
        vibrt_cycles::make_float3(rng.next(), rng.next(), rng.next()), &value, &wi, &pdf, &roughness);
    s.wi = make_float3(wi.x, wi.y, wi.z);
    s.valid = pdf > 0;
    s.lobe = LOBE_GLOSSY;
    s.sampled_value = make_float3(value.x, value.y, value.z);
    s.sampled_pdf = pdf;
    s.weighted_sample = true;
    return s;
  }
#endif
  float NoV = dot3(e.Ns, wo);
  if (NoV <= 0.0f)
    return s;
#if VIBRT_CLOSURE_TYPES
  if (e.mat->closure_type) {
    if (e.mat->closure_type <= 2) {
      auto toon = toon_closure(e);
      auto *sc = reinterpret_cast<const vibrt_cycles::ShaderClosure *>(&toon);
      vibrt_cycles::float3 value = {}, direction = {};
      float pdf;
      auto random = vibrt_cycles::make_float2(rng.next(), rng.next());
      int label =
          e.mat->closure_type == 1
              ? vibrt_cycles::bsdf_diffuse_toon_sample(sc, cycles_vector(e.Ng), cycles_vector(wo),
                                                       random, &value, &direction, &pdf)
              : vibrt_cycles::bsdf_glossy_toon_sample(sc, cycles_vector(e.Ng), cycles_vector(wo),
                                                      random, &value, &direction, &pdf);
      s.wi = make_float3(direction.x, direction.y, direction.z);
      s.valid = label != vibrt_cycles::LABEL_NONE && pdf > 0;
      s.lobe = e.mat->closure_type == 1 ? LOBE_DIFFUSE : LOBE_GLOSSY;
      s.sampled_value = e.base_color * make_float3(value.x, value.y, value.z);
      s.sampled_pdf = pdf;
      s.weighted_sample = true;
    }
    return s;
  }
#endif

  // Cycles Principled `Alpha` socket: `alpha × principled + (1-alpha) ×
  // transparent` blend. With probability (1-alpha), pick the transparent
  // passthrough branch — the ray continues straight through the surface
  // (wi = -wo, the back hemisphere) with full transmission. The path
  // tracer's `throughput *= bs.f / bs.pdf` then preserves throughput on
  // this branch. eval_bsdf scales the opaque branch by alpha so NEE and
  // BSDF-sampled emission see the correct mixing weight.
  float ab = e.alpha_blend;
  if (ab < 1.0f && rng.next() >= ab) {
    s.wi = -1.0f * wo;
    s.transparent = true;
    s.valid = true;
    s.specular = true;
    s.lobe = LOBE_TRANSMISSION;
    return s;
  }

  if (e.mat->hair_weight > 0.5f) {
    auto hair = classic_hair(e);
    vibrt_cycles::float3 value = {}, direction = {};
    vibrt_cycles::float2 roughness;
    float pdf;
    auto random = vibrt_cycles::make_float2(rng.next(), rng.next());
    int label = vibrt_cycles::sample_classic_hair(&hair, e.mat->hair_component != 0,
                                                  cycles_vector(e.Ng), cycles_vector(wo), random,
                                                  &value, &direction, &pdf, &roughness);
    s.wi = make_float3(direction.x, direction.y, direction.z);
    s.valid = pdf > 0;
    s.lobe = label & vibrt_cycles::LABEL_TRANSMIT ? LOBE_TRANSMISSION : LOBE_GLOSSY;
    s.sampled_value = e.base_color * make_float3(value.x, value.y, value.z) * e.alpha_blend;
    s.sampled_pdf = pdf * e.alpha_blend;
    s.weighted_sample = true;
    return s;
  }

  // Coat weight mirrors eval_bsdf: scale by F_coat(V) so sampling budget
  // follows the actual coat intensity at this view angle (grazing >> normal).
  const auto &closures = surface_closures(e, wo, cache);
  vibrt_cycles::float3 direction, value;
  float pdf;
  int label = vibrt_surface::sample(closures, rng.next(),
      vibrt_cycles::make_float3(rng.next(), rng.next(), rng.next()), &direction, &value, &pdf);
  s.wi = make_float3(direction.x, direction.y, direction.z);
  s.valid = label != vibrt_cycles::LABEL_NONE && pdf > 0;
  s.specular = (label & vibrt_cycles::LABEL_SINGULAR) != 0;
  s.lobe = label & vibrt_cycles::LABEL_TRANSMIT ? LOBE_TRANSMISSION
         : label & vibrt_cycles::LABEL_DIFFUSE ? LOBE_DIFFUSE : LOBE_GLOSSY;
  s.sampled_value = make_float3(value.x, value.y, value.z) * e.alpha_blend;
  if (dot3(e.Ns, s.wi) > 0) s.sampled_value = s.sampled_value * bump_shadowing_term(e.Ng, e.Ns, s.wi);
  s.sampled_pdf = pdf * e.alpha_blend;
  s.weighted_sample = true;
  return s;
}

// Multi-closure eval: weighted sum over the leaves emitted by
// `eval_material`. Both f and pdf are linearly weighted; the combined
// pdf is the mixture pdf, which is what the path tracer needs for MIS
// against light pdfs (and what `sample_bsdf` below produces). When
// `g.n == 1` this is just `eval_bsdf_one(g.slots[0], wo, wi)`.
static __device__ BsdfEval eval_bsdf(const MaterialEvalGroup &g,
                                     float3 wo, float3 wi,
                                     SurfaceClosureCache &cache,
                                     int sampled_index = -1) {
  BsdfEval r;
  r.f = make_float3(0.0f, 0.0f, 0.0f);
  r.pdf = 0.0f;
  float total_weight=0.0f;
  for (int i = 0; i < g.n; ++i) {
    BsdfEval ev = eval_bsdf_one(g.slots[i], wo, wi, cache);
    // Preserve rejection of a direction with zero density in the sampled
    // leaf, even when other leaves would give that direction a nonzero PDF.
    if (i == sampled_index && ev.pdf <= 0.0f) {
      r.f = make_float3(0, 0, 0);
      r.pdf = 0.0f;
      return r;
    }
    float w = g.weights[i];
    r.f = r.f + ev.f * w;
    r.pdf += ev.pdf * w;
    total_weight+=w;
  }
  if(total_weight>0)r.pdf/=total_weight;
  return r;
}

// Multi-closure sampler. Picks one leaf by its weight, samples within
// it, then evaluates the combined f / pdf via the `eval_bsdf` wrapper
// so MIS weights against light samplers see the real mixture density.
// `lobe` and `specular` come from the inner sample — they describe
// which closure produced wi for the per-lobe bounce-cap accounting in
// trace_path.
static __device__ BsdfSample sample_bsdf(const MaterialEvalGroup &g,
                                         float3 wo, RNG &rng, SurfaceClosureCache &cache) {
  int picked = 0;
  float total_weight = 1.0f;
  if (g.n > 1) {
    total_weight=0.0f;for(int i=0;i<g.n;i++)total_weight+=g.weights[i];
    float u = rng.next()*total_weight;
    picked = g.n - 1; // last leaf catches residual mass
    float acc = 0.0f;
    for (int i = 0; i < g.n - 1; ++i) {
      acc += g.weights[i];
      if (u < acc) {
        picked = i;
        break;
      }
    }
  }
  BsdfDirectionSample direction = sample_bsdf_direction(g.slots[picked], wo, rng, cache);
  BsdfSample s;
#if (VIBRT_CLOSURE_TYPES & (1u << 4))
  s.portal = g.slots[picked].mat->closure_type == 4;
  if (s.portal) {
    s.portal_position = g.slots[picked].portal_position;
    s.wi = g.slots[picked].portal_direction;
    float w = g.n > 1 ? g.weights[picked] : 1;
    s.f = g.slots[picked].base_color * w;
    s.pdf = w / fmaxf(total_weight, 1e-20f);
    s.specular = true;
    s.transparent = true;
    s.lobe = LOBE_TRANSMISSION;
    return s;
  }
#endif
  s.wi = direction.wi;
  s.f = make_float3(0, 0, 0);
  s.pdf = 0.0f;
  s.specular = direction.specular;
  s.transparent = direction.transparent;
  s.lobe = direction.lobe;
  if (!direction.valid)
    return s;
  if (direction.weighted_sample) {
    // Preserve the selected closure's sampled estimate and density.
    // Only the other mixture leaves need a separate evaluation.
    for (int i = 0; i < g.n; i++) {
      BsdfEval value;
      if (i == picked) {
        value.f = direction.sampled_value;
        value.pdf = direction.sampled_pdf;
      } else
        value = eval_bsdf_one(g.slots[i], wo, s.wi, cache);
      float w = g.n == 1 ? 1 : g.weights[i];
      s.f = s.f + value.f * w;
      s.pdf += value.pdf * w;
    }
    s.pdf /= total_weight;
    return s;
  }
  if (s.transparent) {
    if (g.n == 1) {
      s.f = g.slots[0].mat->closure_type == 8 ? g.slots[0].base_color : make_float3(1, 1, 1);
      s.pdf = 1.0f;
      return s;
    }
    float opacity=0.0f;
    for(int i=0;i<g.n;i++) {
      float weight = g.weights[i] * (1.0f - g.slots[i].alpha_blend);
      opacity += weight;
      s.f = s.f + (g.slots[i].mat->closure_type == 8 ? g.slots[i].base_color : make_float3(1, 1, 1)) * weight;
    }
    s.pdf=opacity/fmaxf(total_weight,1e-20f);
    return s;
  }
  // Evaluate every leaf once, in the original accumulation order. Deferring
  // evaluation avoids both a redundant call and per-lane skips of sampled
  // leaves, which would still run the evaluator for other lanes in the warp.
  BsdfEval comb = g.n == 1 ? eval_bsdf_one(g.slots[0], wo, s.wi, cache)
                          : eval_bsdf(g, wo, s.wi, cache, picked);
  s.f = comb.f;
  s.pdf = comb.pdf;
  return s;
}

static __device__ __noinline__ void fill_surface_vertex(PathVertex*,HitGroupData*,unsigned int,float2,float3,float,unsigned int,const float3 * = nullptr);

struct ShadowCall { const PathVertex *vertex; float3 result; };
static __device__ __noinline__ float3 surface_shadow_transparency_impl(const PathVertex &);
static __device__ __noinline__ float3 surface_shadow_transparency(const PathVertex &v) {
#if (VIBRT_SHADER_OPS & (7ull << 41))
  ShadowCall call = {&v, {}};
  optixContinuationCall<void, ShadowCall *>(4, &call);
  return call.result;
#else
  return surface_shadow_transparency_impl(v);
#endif
}
static __device__ __noinline__ float3 surface_shadow_transparency_impl(const PathVertex &v) {
  struct Entry { PrincipledGpu *material; float weight; bool leaf; };
  Entry stack[32];
  int count = 0;
  stack[count++] = {v.mat, 1, false};
  vibrt_shadow::ShaderData sd{vibrt_shadow::SD_TRANSPARENT, vibrt_cycles::make_float3(0)};
  while (count) {
    Entry entry = stack[--count];
    auto *m = entry.material;
    if (!m || entry.weight == 0)
      continue;
    if (!entry.leaf && (m->left_subtree || m->secondary)) {
      float fac = compute_level_mix_fac(m, v, fabsf(dot3(v.Ns, v.incoming)));
      stack[count++] = {m->secondary ? m->secondary : m, entry.weight * (m->add_shader ? 1 : fac), !m->secondary};
      stack[count++] = {m->left_subtree ? m->left_subtree : m, entry.weight * (m->add_shader ? 1 : 1 - fac), !m->left_subtree};
      continue;
    }
    if (m->closure_type == 4)
      continue; // Cycles excludes ray portals from transparent shadow traversal.
#if (VIBRT_CLOSURE_TYPES & (1u << 6))
    if (m->closure_type == 6) {
      ShaderEvalCache cache{};
      auto hair = eval_material_one(v, m, cache);
      if (hair.hair_context.transparent)
        sd.closure_transparent_extinction += vibrt_cycles::make_float3(entry.weight);
      continue;
    }
#endif
    float alpha = m->alpha_blend;
    float3 color = m->closure_type == 8 ? make_f3(m->base_color) : make_float3(1, 1, 1);
    if (m->closure_type == 8 && m->base_color_tex) {
      const float *M = m->uv_transform;
      float2 uv = make_float2(M[0] * v.uv.x + M[1] * v.uv.y + M[2], M[3] * v.uv.x + M[4] * v.uv.y + M[5]);
      color = color * sample_rgba(m->base_color_tex, m->base_color_tex_w, m->base_color_tex_h, m->base_color_tex_channels, uv);
    }
    if (m->shader_outputs & ((1u << 7) | (m->closure_type == 8 ? 1u : 0u))) {
      vibrt_cycles::float4 out[VIBRT_SHADER_OUTPUT_COUNT];
      evaluate_shader(m, v, out, m->closure_type == 8 ? vibrt_shader::EVAL_ALL : vibrt_shader::EVAL_ALPHA);
      if (m->shader_outputs & (1u << 7)) alpha = __saturatef(out[7].x);
      if (m->closure_type == 8 && (m->shader_outputs & 1u)) color = make_float3(out[0].x, out[0].y, out[0].z);
    }
    sd.closure_transparent_extinction += cycles_vector(color) * (entry.weight * (1 - alpha));
  }
  auto value = vibrt_shadow::surface_shader_transparency(&sd);
  return make_float3(value.x, value.y, value.z);
}

struct EmissionCall { const PathVertex *vertex; bool include_emission; float4 result; };
static __device__ __noinline__ float4 emission_alpha_impl(const PathVertex &, bool);
static __device__ __noinline__ float4 emission_alpha(const PathVertex &v, bool include_emission = true) {
#if (VIBRT_SHADER_OPS & (7ull << 41))
  EmissionCall call = {&v, include_emission, {}};
  optixContinuationCall<void, EmissionCall *>(3, &call);
  return call.result;
#else
  return emission_alpha_impl(v, include_emission);
#endif
}
static __device__ __noinline__ float4 emission_alpha_impl(const PathVertex &v,bool include_emission=true) {
  struct Entry {PrincipledGpu *m;float w;bool leaf;};
  Entry stack[32];int sp=0;stack[sp++]={v.mat,1,false};
  float3 emission=make_float3(0,0,0);float transparent=0;
  while(sp){Entry e=stack[--sp];if(e.w==0.0f)continue;PrincipledGpu *m=e.m;
    if(!e.leaf&&(m->left_subtree||m->secondary)){
      float fac=compute_level_mix_fac(m,v,fabsf(dot3(v.Ns,v.incoming)));
      stack[sp++]={m->secondary?m->secondary:m,e.w*(m->add_shader?1:fac),!m->secondary};
      stack[sp++]={m->left_subtree?m->left_subtree:m,e.w*(m->add_shader?1:1-fac),!m->left_subtree};
      continue;
    }
    // Emissive Principled layers must use the same attenuation for light
    // samples and visible hits; evaluating emission alone skips coat/sheen.
    if (include_emission && !m->surface_model && !m->pure_diffuse && !m->pure_glossy &&
        (m->coat_weight > 0 || m->sheen_weight > 0 ||
         (shader_outputs(m) & ((1ull << 11) | (1ull << 14))))) {
      ShaderEvalCache cache{};
      MaterialEval material = eval_material_one(v, m, cache);
      emission = emission + material.emission * e.w;
      transparent += e.w * (1 - material.alpha_blend);
      continue;
    }
    float3 em=make_f3(m->emission);float alpha=m->alpha_blend;
    if(include_emission&&m->emission_tex){const float *M=m->uv_transform;float2 uv=make_float2(M[0]*v.uv.x+M[1]*v.uv.y+M[2],M[3]*v.uv.x+M[4]*v.uv.y+M[5]);em=em*sample_rgba(m->emission_tex,m->emission_tex_w,m->emission_tex_h,m->emission_tex_channels,uv);}
    unsigned int outputs=(1u<<7)|(include_emission?((1u<<5)|(1u<<6)):0);
    if (shader_outputs(m) & outputs) {
      vibrt_cycles::float4 out[VIBRT_SHADER_OUTPUT_COUNT];
      evaluate_shader(m, v, out,
                      vibrt_shader::EVAL_ALPHA |
                          (include_emission ? vibrt_shader::EVAL_EMISSION : 0));
      if(include_emission&&(shader_outputs(m)&(1u<<5)))em=make_float3(out[5].x,out[5].y,out[5].z);
      if(include_emission&&(shader_outputs(m)&(1u<<6)))em=em*out[6].x;
      if(shader_outputs(m)&(1u<<7))alpha=fminf(fmaxf(out[7].x,0.0f),1.0f);
    }
    emission=emission+em*(e.w*alpha);transparent+=e.w*(1-alpha);
  }
  return make_float4(emission.x,emission.y,emission.z,1-fminf(transparent,1.0f));
}

#if VIBRT_AOVS
static __device__ __noinline__ void write_shader_aovs(const PathVertex &v,
                                                      float coverage) {
  if (!params.aov_data || !v.mat || coverage <= 0)
    return;
  unsigned int pixel =
      optixGetLaunchIndex().y * params.width + optixGetLaunchIndex().x;
  unsigned int count = params.width * params.height;
  for (unsigned int i = 0; i < v.mat->num_aov_shaders; i++) {
    const AovShader &a = v.mat->aov_shaders[i];
    if (a.index >= params.num_aovs)
      continue;
    vibrt_cycles::float4 out[VIBRT_SHADER_OUTPUT_COUNT];
    evaluate_shader(a.material, v, out);
    float *dest = params.aov_data + 4 * (a.index * count + pixel);
    for (int ch = 0; ch < 3; ch++)
      dest[ch] += out[0][ch] * coverage;
    dest[3] += coverage;
  }
}
#endif
static __device__ float3 mesh_light_emission(const MeshLightTri &tri,float b1,float b2,float3 incoming,float distance) {
  PathVertex v;fill_surface_vertex(&v,tri.geometry,tri.primitive,make_float2(b1,b2),incoming,distance,2u);
  float4 e=emission_alpha(v);return make_float3(e.x,e.y,e.z);
}

// NEE-only finite point/spot lights. Uniform solid-angle sampling of a sphere
// (or the legacy facing disk) includes the radiance/PDF conversion in weight.
// As radius tends to zero this converges to the delta intensity / distance².
static __device__ float sample_point_position(float3 center, float radius,
                                              unsigned int sphere, float3 P,
                                              RNG &rng, float3 &position,
                                              float3 &wi, float &distance) {
  float d2 = dot3(center - P, center - P);
  if (radius <= 1e-6f) {
    distance = sqrtf(d2);
    position = center;
    wi = normalize3(center - P);
    return d2 > 1e-12f ? 1 / d2 : 0;
  }
  // A sphere center has no orientation; the upstream exterior formula divides
  // by distance. Move only this singular evaluation point by a negligible offset.
  auto point = cycles_vector(d2 > 0 ? P : P + make_float3(0, 0, radius * 1e-7f));
  vibrt_lights::KernelLight light = {};
  light.co = cycles_vector(center);
  light.spot.radius = radius;
  light.spot.is_sphere = sphere != 0;
  light.spot.eval_fac = 1 / (M_PIf * radius * radius);
  vibrt_lights::LightSample sample = {};
  vibrt_lights::point_light_sample(&light, vibrt_cycles::make_float2(rng.next(), rng.next()),
                                   point, vibrt_cycles::make_float3(0, 0, 1),
                                   vibrt_cycles::SD_BSDF_HAS_TRANSMISSION, &sample);
  position = make_float3(sample.P.x, sample.P.y, sample.P.z);
  wi = make_float3(sample.D.x, sample.D.y, sample.D.z);
  distance = sample.t;
  return sample.pdf > 0 ? sample.eval_fac / sample.pdf : 0;
}

static __device__ float sample_spot_position(const SpotLight &sp, float3 P, RNG &rng,
                                             float3 &position, float3 &wi, float &distance) {
  vibrt_lights::KernelLight light = {};
  light.co = cycles_vector(make_f3(sp.position));
  light.spot.radius = sp.radius;
  light.spot.is_sphere = sp.sphere != 0;
  light.spot.dir = cycles_vector(make_f3(sp.direction));
  light.spot.cos_half_spot_angle = sp.cos_outer;
  light.spot.cos_half_larger_spread = sp.cos_larger_spread;
  light.inverse = {cycles_vector(make_f3(sp.world_to_light)),
                   cycles_vector(make_f3(sp.world_to_light + 3)),
                   cycles_vector(make_f3(sp.world_to_light + 6))};
  light.spot.spot_smooth = 1 / fmaxf(sp.cos_inner - sp.cos_outer, 1e-20f);
  if (sp.radius <= 1e-6f) {
    float weight = sample_point_position(make_f3(sp.position), 0, sp.sphere, P, rng,
                                          position, wi, distance);
    auto local = vibrt_lights::spot_light_to_local(0, &light, cycles_vector(-wi));
    return weight * vibrt_lights::spot_light_attenuation(&light.spot, local);
  }
  light.spot.eval_fac = 1 / (M_PIf * sp.radius * sp.radius);
  vibrt_lights::LightSample sample = {};
  auto p = cycles_vector(P);
  if (vibrt_cycles::len_squared(p - light.co) == 0)
    p.z += sp.radius * 1e-7f;
  bool ok = vibrt_lights::spot_light_sample<false>(
      0, &light, vibrt_cycles::make_float2(rng.next(), rng.next()), p,
      vibrt_cycles::make_float3(0, 0, 1), vibrt_cycles::SD_BSDF_HAS_TRANSMISSION, &sample);
  position = make_float3(sample.P.x, sample.P.y, sample.P.z);
  wi = make_float3(sample.D.x, sample.D.y, sample.D.z);
  distance = sample.t;
  return ok && sample.pdf > 0 ? sample.eval_fac / sample.pdf : 0;
}

static __device__ float3 sample_sun_direction(const SunLight &sun, RNG &rng, float &pdf) {
  vibrt_lights::KernelLight light = {};
  light.co = cycles_vector(-make_f3(sun.direction));
  light.distant.one_minus_cosangle = 1 - sun.cos_angle;
  vibrt_lights::LightSample sample = {};
  vibrt_lights::distant_light_sample(&light, vibrt_cycles::make_float2(rng.next(), rng.next()), &sample);
  pdf = sample.pdf;
  return make_float3(sample.D.x, sample.D.y, sample.D.z);
}

static __device__ float3 point_light_emission(PrincipledGpu *shader, float3 value,
                                              float3 P, float3 incoming, float distance) {
  if (!shader) return value;
  PathVertex v = {};
  v.mat = shader;
  v.P = P;
  v.Ns = v.Ng = v.incoming = incoming;
  build_frame(v.Ns, v.T, v.dPdv);
  v.ray_length = distance;
  v.path_flags = 2u;
  float4 evaluated = emission_alpha(v);
  return value * make_float3(evaluated.x, evaluated.y, evaluated.z);
}

static __device__ float3 area_light_emission(const AreaRectLight &light, float3 P,
                                             float3 incoming, float distance,
                                             unsigned int path_flags = 2u) {
  float3 value = make_f3(light.emission);
  if (light.normalize_spread > 0) {
    float3 normal = make_f3(light.normal);
    if (light.two_sided && dot3(normal, incoming) < 0)
      normal = -normal;
    value = value * vibrt_lights::area_light_spread_attenuation(
                        cycles_vector(-incoming), cycles_vector(normal),
                        light.tan_half_spread, light.normalize_spread);
  }
  if (light.shader) {
    PathVertex v = {};
    v.mat = light.shader;
    v.P = P;
    v.Ns = v.Ng = make_f3(light.normal);
    v.T = make_f3(light.u_axis);
    v.incoming = incoming;
    v.ray_length = distance;
    v.path_flags = path_flags;
    float3 delta = P - make_f3(light.corner);
    v.uv = make_float2(dot3(delta, make_f3(light.u_axis)) / light.size_u,
                      dot3(delta, make_f3(light.v_axis)) / light.size_v);
    float4 evaluated = emission_alpha(v);
    value = value * make_float3(evaluated.x, evaluated.y, evaluated.z);
  }
  return value;
}
