// ---------- Colour graph evaluator ----------
#include "vibrt_nodes_core.h"
#include "vibrt_shader.h"
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
#define NEED(I) (mask & (1u << (I)))
#define CTX_VEC(I,V) if(NEED(I))ctx[I]=make_float4((V).x,(V).y,(V).z,1)
  constexpr float delta = 0.001f;
  if(NEED(0))ctx[0]=make_float4(v.uv.x+(derivative==1?delta:0),v.uv.y+(derivative==2?delta:0),0,1);
  if(NEED(1)) {
    float3 G=v.generated;
    if(derivative==1)G=G+v.dGdu*delta;else if(derivative==2)G=G+v.dGdv*delta;
    CTX_VEC(1,G);
  }
  if(NEED(2)) {
    float3 O=v.P_object;
    if(derivative==1)O=O+v.dOdu*delta;else if(derivative==2)O=O+v.dOdv*delta;
    CTX_VEC(2,O);
  }
  float3 P=v.P;
  if(mask&((1u<<3)|(1u<<12)|(1u<<14))) {
    if(derivative==1)P=P+v.dPdu*delta;else if(derivative==2)P=P+v.dPdv*delta;
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
static __device__ __noinline__ void evaluate_shader_vm(PrincipledGpu *m,const PathVertex &v,vibrt_cycles::float4 *out,unsigned int requested,int derivative) {
  vibrt_cycles::float4 ctx[32];shader_context(v,ctx,m->shader_contexts,derivative);
  vibrt_shader::SceneServices services{&v, derivative};
  vibrt_shader::evaluate_vm((const ShaderInstruction *)m->shader_nodes,
                            m->shader_len, ctx, out, requested, &services);
}
static __device__ __noinline__ void evaluate_shader_single(PrincipledGpu *m,float2 uv,vibrt_cycles::float4 *out,unsigned int requested) {
  const ShaderInstruction &n=*(const ShaderInstruction*)m->shader_nodes;
  if ((n.op >> 8) & requested)
    vibrt_shader::evaluate_single(n,vibrt_cycles::make_float4(uv.x,uv.y,0,1),out);
}
static __device__ __forceinline__ void evaluate_shader(PrincipledGpu *m,const PathVertex &v,vibrt_cycles::float4 *out,unsigned int requested=vibrt_shader::EVAL_ALL,int derivative=0) {
  if (m->shader_len == 1) {
    float2 uv=v.uv;
    if(derivative==1)uv.x+=0.001f;else if(derivative==2)uv.y+=0.001f;
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
  vibrt_cycles::ShaderData hair_context;
#endif
  PrincipledGpu *mat; // raw material ptr for coat/sheen/sss params
};

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
    unsigned long long mask = m->shader_outputs;
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
#if (VIBRT_CLOSURE_TYPES & (1u << 7))
    GRAPH_SCALAR(41, sss_scale);
    GRAPH_SCALAR(42, sss_ior);
    GRAPH_SCALAR(43, sss_anisotropy);
#endif
#if (VIBRT_CLOSURE_TYPES & (1u << 5))
    if (m->closure_type == 5) {
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
      e.hair_context.type =
          v.geometry->tangents ? vibrt_cycles::PRIMITIVE_CURVE_RIBBON : 0;
      e.hair_context.v = v.uv.x * 2 - 1;
      e.chiang = vibrt_shader::chiang_setup(
          e.hair_context,
          vibrt_cycles::make_float3(e.base_color.x, e.base_color.y,
                                    e.base_color.z),
          vibrt_shader::xyz(out[44]), e.hair_roughness_u, e.hair_roughness_v,
          e.hair_offset, out[47].x, out[46].x, out[45].x, out[52].x, out[53].x,
          out[54].x, out[55].x, vibrt_shader::xyz(out[56]), (int)out[57].x);
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
      float3 r1=cross3(v.dPdv,graph_normal),r2=cross3(graph_normal,v.dPdu);
      float det=dot3(v.dPdu,r1);
      if(fabsf(det)>1e-12f){float3 bumped=normalize3(graph_normal-(r1*dhdu+r2*dhdv)*(out[28].x/det));float f=fminf(fmaxf(out[27].x,0.0f),1.0f);graph_normal=normalize3(graph_normal*(1-f)+bumped*f);}
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
  e.alpha = fmaxf(e.roughness * e.roughness, 1e-4f);
  // Filter-glossy: the path tracer raises `min_alpha` once it's already
  // taken at least one glossy / specular bounce so that subsequent
  // near-mirror evals can't refocus the radiance into a tiny solid
  // angle. Approximates Cycles' Light Paths > Filter Glossy. min_alpha=0
  // (primary ray) leaves alpha untouched.
  if (min_alpha > e.alpha)
    e.alpha = min_alpha;

  float3 Ns = graph_normal;
  // Composite tangent-space perturbation: start from (0,0,1), apply bump
  // (central-difference over heightmap) first, then normal map (multiplied).
  float3 nm_t = make_float3(0.0f, 0.0f, 1.0f);
  bool any_perturb = false;
  if (m->bump_tex != nullptr && m->bump_strength != 0.0f) {
    int bw = m->bump_tex_w;
    int bh = m->bump_tex_h;
    if (bw > 0 && bh > 0) {
      float du = 1.0f / (float)bw;
      float dv = 1.0f / (float)bh;
      float h_px = sample_rgba(m->bump_tex, bw, bh, m->bump_tex_channels,
                               make_float2(uv.x + du, uv.y)).x;
      float h_mx = sample_rgba(m->bump_tex, bw, bh, m->bump_tex_channels,
                               make_float2(uv.x - du, uv.y)).x;
      float h_py = sample_rgba(m->bump_tex, bw, bh, m->bump_tex_channels,
                               make_float2(uv.x, uv.y + dv)).x;
      float h_my = sample_rgba(m->bump_tex, bw, bh, m->bump_tex_channels,
                               make_float2(uv.x, uv.y - dv)).x;
      // Central-difference slope × user strength. Linear in bump_strength
      // so Strength=1 reproduces Blender's default amplitude and the user
      // doesn't need to pre-scale for an empirical "subtle" factor.
      float sx = -(h_px - h_mx) * m->bump_strength;
      float sy = -(h_py - h_my) * m->bump_strength;
      nm_t = normalize3(make_float3(sx, sy, 1.0f));
      any_perturb = true;
    }
  }
  if (m->normal_tex != nullptr) {
    float3 n = sample_rgba(m->normal_tex, m->normal_tex_w, m->normal_tex_h,
                           m->normal_tex_channels, uv);
    float s = m->normal_strength;
    float3 nn = make_float3((n.x * 2.0f - 1.0f) * s,
                            (n.y * 2.0f - 1.0f) * s,
                             n.z * 2.0f - 1.0f);
    nn = normalize3(nn);
    // Compose: treat nm_t as frame, re-express nn in its tangent basis.
    if (any_perturb) {
      float3 nt, nb;
      build_frame(nm_t, nt, nb);
      nm_t = normalize3(nt * nn.x + nb * nn.y + nm_t * nn.z);
    } else {
      nm_t = nn;
    }
    any_perturb = true;
  }
  if (any_perturb) {
    float3 T, B;
    build_frame(Ns, T, B);
    Ns = normalize3(T * nm_t.x + B * nm_t.y + Ns * nm_t.z);
  }
  e.Ns = Ns;
  e.Ng = v.Ng;
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

  // Anisotropic α. Disney remap: aspect = sqrt(1 - 0.9·|aniso|).
  float aniso = fminf(fmaxf(e.anisotropy, -1.0f), 1.0f);
  float aspect = sqrtf(fmaxf(1.0f - 0.9f * fabsf(aniso), 1e-4f));
  if (aniso >= 0.0f) {
    e.alpha_x = e.alpha / aspect;
    e.alpha_y = e.alpha * aspect;
  } else {
    e.alpha_x = e.alpha * aspect;
    e.alpha_y = e.alpha / aspect;
  }
  // Coat α also gets the filter-glossy floor applied. Pebble materials
  // in pabellon use a Fresnel-driven MixShader → my exporter routes the
  // glossy side onto the coat lobe with a low coat_roughness (~0.45),
  // which without inflation can refocus indirect bounces through the
  // pool's surface into a sharper cone than the path has already
  // sampled. Same min_alpha as the main lobe — Cycles' filter_glossy
  // is a single state shared across all glossy lobes.
  e.coat_alpha = fmaxf(e.coat_roughness * e.coat_roughness, 1e-4f);
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
  if (m->shader_outputs & (1u << 23)) {
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
    return fminf(fmaxf(fresnel_dielectric(fabsf(NoV), ior), 0.0f), 1.0f);
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
static __device__ __noinline__ int eval_material(const PathVertex &v,
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

// Kajiya-Kay hair: simplified tangent-aligned specular + "sin(T,L)" diffuse.
// Not energy-conservative by construction — it's a classic approximation. The
// specular uses the tilt-shifted H direction so the highlight cones forward
// along the strand (Offset slider on Cycles' Hair BSDF). Sampling is plain
// cosine-weighted hemisphere; MIS picks up the slack.
static __device__ BsdfEval eval_hair(const MaterialEval &e, float3 wo,
                                     float3 wi) {
  BsdfEval r;
  r.f = make_float3(0, 0, 0);
  r.pdf = 0.0f;
  float NoL = dot3(e.Ns, wi);
  if (NoL <= 0.0f)
    return r;

  // Diffuse: peaks when light is perpendicular to the strand.
  float TdotL = dot3(wi, e.T);
  float sin_TL = sqrtf(fmaxf(1.0f - TdotL * TdotL, 0.0f));

  // Specular: tangent-aligned cone. sin(offset) tilts the cone along T so the
  // highlight shifts toward the tip of the strand (matches Cycles' "Offset").
  float3 H = wo + wi;
  float Hlen = sqrtf(fmaxf(dot3(H, H), 1e-12f));
  H = H * (1.0f / Hlen);
  float TdotH = dot3(H, e.T);
  float tilt = sinf(e.hair_offset);
  float TdotH_eff = TdotH - tilt;
  float cone = sqrtf(fmaxf(1.0f - TdotH_eff * TdotH_eff, 0.0f));
  float exponent = 2.0f / fmaxf(e.hair_roughness_u *
                                    e.hair_roughness_u,
                                1e-4f);
  float spec = powf(fmaxf(cone, 0.0f), exponent);

  r.f = (e.base_color * sin_TL + make_float3(spec, spec, spec))
        * (INV_PIf * NoL);
  r.pdf = NoL * INV_PIf;
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
                                         float3 wi) {
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

  float NoV = dot3(e.Ns, wo);
  if (NoV <= 0.0f)
    return r;
#if VIBRT_CLOSURE_TYPES
  if (e.mat->closure_type) {
    if (e.mat->closure_type <= 2 && dot3(e.Ns, wi) > 0) {
      float3 axis = e.mat->closure_type == 1 ? e.Ns : e.Ns * (2 * NoV) - wo;
      auto value =
          vibrt_shader::toon_eval(dot3(axis, wi), e.toon_size, e.toon_smooth);
      r.f = e.base_color * value.x;
      r.pdf = value.y;
    }
    return r;
  }
#endif
  if (e.mat->hair_weight > 0.5f)
    return eval_hair(e, wo, wi);

  float NoL = dot3(e.Ns, wi);
  bool reflect = NoL > 0.0f;
  bool transmit = !reflect && e.transmission > 0.0f;
  // Translucent BSDF: Lambertian on the back hemisphere. Active when wi is
  // on the side opposite wo (NoL < 0) — light passing through a thin sheet.
  bool translucent_active =
      !reflect && e.translucent_weight > 0.0f && e.transmission <= 0.0f;

  // Cycles Principled F0 = ((ior-1)/(ior+1))² × 2 × specular_ior_level.
  // The user-set Specular IOR Level (default 0.5 → multiplier 1.0)
  // scales F0 — flat_archiviz's Anta Cucina / Base / Book materials
  // (Specular IOR Level 0.30-0.35) pushed the dielectric Schlick lift
  // by 30-40% over Cycles before this fix.
  float F0_d =
      ((e.ior - 1.0f) / (e.ior + 1.0f)) * ((e.ior - 1.0f) / (e.ior + 1.0f))
      * (2.0f * e.specular_ior_level);

  // Coat Fresnel parameters (used both for MIS weighting and the coat BRDF).
  PrincipledGpu *mc = e.mat;
  float coat_w_mat = e.coat_weight;
  float coat_alpha = e.coat_alpha;  // already filter-glossy-clamped
  float cF0 = ((e.coat_ior - 1.0f) / (e.coat_ior + 1.0f)) *
              ((e.coat_ior - 1.0f) / (e.coat_ior + 1.0f));

  // Weights for MIS between lobes. Coat gets `coat_weight·F_coat(V)` so its
  // sampling budget follows the actual lobe intensity at this view angle —
  // at grazing the coat dominates, at normal it contributes weakly.
  // translucent_weight steals from the forward diffuse budget: t_w=1 → no
  // forward Lambert, all diffuse energy on the back hemisphere; t_w=0.5 →
  // even split (Mix Shader of Diffuse + Translucent).
  float t_w = e.translucent_weight;
  float w_base_diff = (1.0f - e.metallic) * (1.0f - e.transmission);
  float w_diffuse = w_base_diff * (1.0f - t_w) * (1.0f - e.sss_weight);
  float w_translucent = w_base_diff * t_w;
  // Cycles' Diffuse BSDF has no specular / coat layer. Force the
  // associated lobes to zero so we end up with exactly one Lambertian
  // term — the existing Principled mapping leaks a `(1-VoH)^5 × GGX`
  // spec contribution at grazing even with metallic=0, which over-
  // brightens scenes dominated by ShaderNodeBsdfDiffuse (BMW27's car
  // body).
  bool pure_diff = (mc->pure_diffuse != 0);
  float w_spec = pure_diff ? 0.0f
      : e.metallic + (1.0f - e.metallic) * (1.0f - e.transmission);
  float w_trans = pure_diff ? 0.0f
      : (1.0f - e.metallic) * e.transmission;
  float w_coat = pure_diff ? 0.0f
      : coat_w_mat * schlick_scalar(NoV, cF0);
  float w_sum = w_diffuse + w_translucent + w_spec + w_trans + w_coat;
  if (w_sum <= 0.0f)
    return r;
  float p_diff = w_diffuse / w_sum;
  float p_translucent = w_translucent / w_sum;
  float p_spec = w_spec / w_sum;
  float p_trans = w_trans / w_sum;
  float p_coat = w_coat / w_sum;

  float NoL_std = fmaxf(NoL, 0.0f);

  if (reflect) {
    // The remaining local diffuse component; SSS is sampled separately.
    if (w_diffuse > 0.0f) {
      float effective = NoL_std;
      // Energy coupling with the dielectric specular layer above the
      // diffuse. Cycles' Principled attenuates the diffuse by the
      // specular layer's directional-hemispherical reflectance so the
      // pair conserves energy: a white-furnace base-0.8 plane reads
      // albedo 0.750 (= 0.8 * (1 - E_spec)) in Cycles, but vibrt read
      // 0.803 (the full Lambert) — an unattenuated diffuse that, summed
      // over multi-bounce GI, ran the closed 0.8-box ~1.18x over. The
      // attenuation is VIEW-side only (E_spec(NoV), integrated over
      // incoming) — the earlier directional (1-F(NoV))(1-F(NoL)) form
      // over-darkened grazing-lit direct shading. `pure_diff` (Cycles'
      // standalone Diffuse BSDF, no spec layer) keeps the full Lambert.
      float spec_T = 1.0f;
      if (!pure_diff) {
        // Diffuse-under-dielectric energy coupling: reduce the diffuse by
        // the specular layer's directional-hemispherical reflectance. The
        // reflectance scale is F0 (the normal-incidence dielectric Fresnel),
        // NOT the Schlick hemispherical average f_avg_schlick(F0) = (20F0+1)/21
        // which for F0=0.04 gives 0.086 — ~2× too high (the (1/21) term is the
        // grazing F90→1 lift, which the ggx_e_lookup directional factor already
        // accounts for). Using f_avg over-reduced the diffuse → ~4% under on
        // Principled-dominated scenes (flat_archiviz / pabellon / bmw27).
        float E_spec_v = F0_d * ggx_e_lookup(NoV, e.alpha) +
                         ggx_b_lookup(NoV, e.alpha) * (1.0f - F0_d);
        spec_T = 1.0f - fminf(fmaxf(E_spec_v, 0.0f), 0.99f);
      }
      r.f = r.f + e.base_color * (INV_PIf * w_diffuse * effective * spec_T);
      r.pdf += p_diff * NoL_std * INV_PIf;
    }
    // Specular (metallic + dielectric spec layer). Skipped entirely for
    // pure-Lambertian Diffuse mappings — Cycles' Diffuse BSDF has no
    // specular layer and the residual `(1-VoH)^5 × GGX` from the
    // dielectric Schlick lifts diffuse materials toward white at
    // glancing angles (the BMW27 car body went +0.10 brighter than
    // Cycles before this gate).
    if (!pure_diff) {
    float3 H = normalize3(wo + wi);
    float NoH = fmaxf(dot3(e.Ns, H), 0.0f);
    float VoH = fmaxf(dot3(wo, H), 0.0f);
    float3 Vloc = make_float3(dot3(wo, e.T), dot3(wo, e.B), dot3(wo, e.Ns));
    float3 Lloc = make_float3(dot3(wi, e.T), dot3(wi, e.B), dot3(wi, e.Ns));
    float3 Hloc = make_float3(dot3(H, e.T), dot3(H, e.B), dot3(H, e.Ns));
    float D = ggx_D_aniso(Hloc, e.alpha_x, e.alpha_y);
    float G1_v = smith_G1_aniso(Vloc, e.alpha_x, e.alpha_y);
    float G1_l = smith_G1_aniso(Lloc, e.alpha_x, e.alpha_y);
    float G = G1_v * G1_l;
    // For Cycles' Glossy / Anisotropic BSDF the GGX lobe carries no
    // Fresnel — `bsdf_microfacet_ggx_setup` sets MicrofacetFresnel::NONE
    // and the per-direction reflectance is just `Color × D × G / (4·NoV·NoL)`.
    // `pure_glossy` short-circuits Schlick to a uniform `base_color` tint;
    // without this the spec lobe lifts toward white at grazing on dark
    // metallic mappings (BMWBlack on bmw27 was visibly too bright on the
    // rounded car body). Default Principled metal still uses Schlick.
    float3 F_metal = (mc->pure_glossy != 0)
        ? e.base_color
        : schlick_rgb(VoH, e.base_color);
    // Glass / transmissive materials sample the reflect-vs-refract split via
    // `fresnel_dielectric` (exact), so the eval has to use the same Fresnel
    // function — Schlick over-predicts F by ~2× at low IOR (1.0–1.2), which
    // for water (IOR=1.1) at the camera's grazing pool angle made
    // glass-reflection paths carry too much radiance and transmission paths
    // too little. The two formulas match within ~1% for IOR≥1.4 (the typical
    // dielectric range), so non-glass materials don't notice.
    float F_dielec = (e.transmission > 0.0f)
        ? fresnel_dielectric(VoH, e.ior)
        : schlick_scalar(VoH, F0_d);
    float3 f_metal = F_metal * (D * G / fmaxf(4.0f * NoV * NoL, 1e-8f));
    float3 f_dielec = make_float3(F_dielec, F_dielec, F_dielec) *
                      (D * G / fmaxf(4.0f * NoV * NoL, 1e-8f));
    // Dielectric surface reflection. For a matte material this is the
    // standard "spec layer on top of diffuse"; for a glass material
    // (transmission=1) it's the Fresnel-reflective component the
    // transmission lobe's sampler picked when `rng.next() < F`. Without
    // this term, a Glass BSDF surface evaluated on a reflected wi
    // returns f=0 and pdf=0, the path tracer drops the reflection
    // entirely, and at grazing angles where glass should mirror-reflect
    // the rays just transmit straight through to the sky — pabellon's
    // water plane appeared as the bright sunset sky shining through,
    // not as a reflective pond.
    float3 f_spec = e.metallic * f_metal +
                    (1.0f - e.metallic) * f_dielec;

    // Multi-scattering energy compensation — Cycles' exact form
    // (kernel/closure/bsdf_microfacet.h `microfacet_ggx_preserve_energy`).
    // Cycles applies `energy_scale = 1 + (1−E)/E` in the BSDF eval and
    // `darkening = (1 + Fms·(1−E)/E) / energy_scale` in the weight; the
    // product is exactly `1 + Fms·(1−E)/E`, a MULTIPLIER on the single-
    // scatter lobe, with Fms = Fss·Eavg/(1 − Fss·(1−Eavg)) and Fss the
    // LINEAR average reflectance (F_avg, not F_avg²). The previous additive
    // Kulla-Conty form (F_avg² + a separate diffuse-shaped lobe) is a
    // different approximation; matching Cycles' exact algorithm here.
    float E_wo = ggx_e_lookup(NoV, e.alpha);
    float E_avg = ggx_e_avg_lookup(e.alpha);
    float missing = (1.0f - E_wo) / fmaxf(E_wo, 1e-4f);
    float3 one3 = make_float3(1.0f, 1.0f, 1.0f);

    // pure_glossy: F is constant base_color across all angles, so Fss is
    // just base_color (skip the Schlick-averaged form's F90→1 lift).
    float3 Fss_m = (mc->pure_glossy != 0)
        ? e.base_color
        : f_avg_schlick(e.base_color);
    float3 Fms_m = make_float3(
        Fss_m.x * E_avg / fmaxf(1.0f - Fss_m.x * (1.0f - E_avg), 1e-4f),
        Fss_m.y * E_avg / fmaxf(1.0f - Fss_m.y * (1.0f - E_avg), 1e-4f),
        Fss_m.z * E_avg / fmaxf(1.0f - Fss_m.z * (1.0f - E_avg), 1e-4f));
    float3 scale_m = one3 + Fms_m * missing;

    float Fss_d = f_avg_schlick(F0_d);
    float Fms_d = Fss_d * E_avg / fmaxf(1.0f - Fss_d * (1.0f - E_avg), 1e-4f);
    float scale_d = 1.0f + Fms_d * missing;

    f_spec = e.metallic * f_metal * scale_m +
             (1.0f - e.metallic) * f_dielec * scale_d;

    r.f = r.f + f_spec * NoL;
    float pdf_spec = ggx_vndf_pdf_aniso(Vloc, Hloc, e.alpha_x, e.alpha_y);
    r.pdf += p_spec * pdf_spec;
    // Glass reflection samples this same Fresnel lobe via the
    // transmission branch's `rng.next() < F` decision. Account for that
    // contribution so MIS-weighted estimators get the right pdf when
    // we land on a reflected wi for a transmissive surface.
    if (e.transmission > 0.0f && p_trans > 0.0f) {
        r.pdf += p_trans * F_dielec * pdf_spec;
    }

    // --- Coat (clearcoat): additive isotropic GGX dielectric above base. ---
    if (coat_w_mat > 0.0f) {
      float Fc = schlick_scalar(VoH, cF0);
      float cD = ggx_D(NoH, coat_alpha);
      float cG = smith_G1(NoV, coat_alpha) * smith_G1(NoL, coat_alpha);
      float coat_f = coat_w_mat * Fc * cD * cG / fmaxf(4.0f * NoV * NoL, 1e-8f);
      // Round-trip coat transmission attenuates base lobes: (1-F) on entry
      // and on exit.
      float T_view = 1.0f - coat_w_mat * schlick_scalar(NoV, cF0);
      float T_light = 1.0f - coat_w_mat * schlick_scalar(NoL, cF0);
      r.f = r.f * (T_view * T_light);
      r.f = r.f + make_float3(coat_f, coat_f, coat_f) * NoL;
      // Isotropic GGX VNDF density for the coat-sampler. Narrow car-paint
      // clearcoats (coat_roughness ≈ 0.1) are hard for the broad base-spec
      // sampler to hit, so without this branch coat contributions arrive
      // almost entirely through NEE and highlights from environment /
      // indirect bounces fall several-× short of Cycles.
      float pdf_coat = cD * smith_G1(NoV, coat_alpha) /
                       fmaxf(4.0f * NoV, 1e-12f);
      r.pdf += p_coat * pdf_coat;
    }
    } // !pure_diff

    // --- Sheen: soft grazing-angle lobe on top of diffuse. Symmetric in
    // NoV and NoL so the BRDF is reciprocal, and normalised by 1/π so the
    // integral stays bounded as sheen_roughness→0. Still an ad-hoc shape
    // (not Charlie), but energy-sane and reciprocal.
    float sheen_w = e.sheen_weight;
    if (sheen_w > 0.0f) {
      float sr = fmaxf(e.sheen_roughness, 1e-3f);
      float inv_sr = 1.0f / sr;
      float sf = powf(fmaxf(1.0f - NoV, 0.0f), inv_sr) *
                 powf(fmaxf(1.0f - NoL, 0.0f), inv_sr) * sheen_w * INV_PIf;
      float3 tint = e.sheen_tint;
      r.f = r.f + tint * sf * NoL;
    }
  } else if (transmit) {
    // Rough dielectric transmission (Walter et al.). Half-vector
    // h ∝ η_i·ωo + η_t·ωi → divide by η_i, so with eta = η_t/η_i the form is
    // `wo + wi * eta`. This matches the H the VNDF sampler implicitly used
    // in sample_bsdf, so eval's pdf can be inverted back to sampler's pdf.
    float eta = e.ior;
    float3 n_oriented = NoV > 0.0f ? e.Ns : -e.Ns;
    if (NoV < 0.0f)
      eta = 1.0f / eta;
    float3 H = -normalize3(wo + wi * eta);
    if (dot3(H, n_oriented) < 0.0f)
      H = -H;
    float VoH = dot3(wo, H);
    float LoH = dot3(wi, H);
    float abs_VoH = fabsf(VoH);
    float abs_LoH = fabsf(LoH);
    float abs_NoH = fabsf(dot3(e.Ns, H));
    float abs_NoV = fabsf(NoV);
    float abs_NoL = fabsf(NoL);
    float F = fresnel_dielectric(VoH, eta);
    float D = ggx_D(abs_NoH, e.alpha);
    float G1_v = smith_G1(abs_NoV, e.alpha);
    float G = G1_v * smith_G1(abs_NoL, e.alpha);
    float sqrt_den = VoH + eta * LoH;
    float den2 = fmaxf(sqrt_den * sqrt_den, 1e-8f);
    float btdf = (1.0f - F) * D * G * eta * eta * abs_VoH * abs_LoH /
                 (abs_NoV * abs_NoL * den2);
    r.f = e.base_color * btdf * abs_NoL * w_trans;
    // pdf_wi = p_trans · P(refract branch) · VNDF(H) · |dH/dwi|.
    float pdf_h = D * G1_v * abs_VoH / fmaxf(abs_NoV, 1e-12f);
    float jacobian = eta * eta * abs_LoH / den2;
    r.pdf += p_trans * (1.0f - F) * pdf_h * jacobian;
  } else if (translucent_active) {
    // Back hemisphere (NoL < 0). Refraction is mutually
    // exclusive (handled above) — Glass + Translucent isn't a Cycles combo.
    if (translucent_active && w_translucent > 0.0f) {
      float abs_NoL = -NoL;
      r.f = r.f + e.base_color * (INV_PIf * w_translucent * abs_NoL);
      // sample_bsdf uses a cosine-weighted *back* hemisphere, so the pdf
      // matches the forward-Lambert form with |NoL|.
      r.pdf += p_translucent * abs_NoL * INV_PIf;
    }
  }
  // Shadow-terminator: applies to reflection lobes only (the geometric
  // horizon check is for same-side scattering). Transmission / translucent
  // (wi on the far side) keep their own geometry handling.
  if (reflect) {
    r.f = r.f * bump_shadowing_term(e.Ng, e.Ns, wi);
  }
  // Cycles Principled `Alpha` socket: `alpha × principled + (1-alpha) ×
  // transparent` blend. The transparent component is a delta in the
  // through-surface direction, so it contributes 0 to eval at any finite
  // wi — we just scale the opaque part by alpha. sample_bsdf handles
  // the (1-alpha) passthrough branch separately.
  float ab = e.alpha_blend;
  if (ab < 1.0f) {
    r.f = r.f * ab;
    r.pdf *= ab;
  }
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
#if (VIBRT_CLOSURE_TYPES & (1u << 5))
  float3 sampled_value;
  float sampled_pdf;
  bool weighted_sample;
#endif
  float3 wi;
  bool valid;
  bool specular;
  bool transparent;
  int lobe;
};

// Keep direction sampling separate from evaluation so the large sampler's
// call frame is no longer live while eval_bsdf_one runs.
static __device__ __noinline__ BsdfDirectionSample sample_bsdf_direction(const MaterialEval &e, float3 wo,
                                              RNG &rng) {
  BsdfDirectionSample s;
  s.wi = make_float3(0, 0, 1);
  s.valid = false;
  s.specular = false;
  s.lobe = LOBE_DIFFUSE;
  s.transparent = false;
#if (VIBRT_CLOSURE_TYPES & (1u << 5))
  s.weighted_sample = false;
#endif
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

  float NoV = dot3(e.Ns, wo);
  if (NoV <= 0.0f)
    return s;
#if VIBRT_CLOSURE_TYPES
  if (e.mat->closure_type) {
    if (e.mat->closure_type <= 2) {
      float angle = fminf((fminf(fmaxf(e.toon_size, 1e-5f), 1.0f) +
                           __saturatef(e.toon_smooth)) *
                              (0.5f * M_PIf),
                          0.5f * M_PIf);
      float half_sin = sinf(0.5f * angle),
            cosine = 1 - rng.next() * (2 * half_sin * half_sin);
      float sine = sqrtf(fmaxf(1 - cosine * cosine, 0.0f)),
            phi = 2 * M_PIf * rng.next();
      float3 axis = e.mat->closure_type == 1 ? e.Ns : e.Ns * (2 * NoV) - wo, T,
             B;
      build_frame(axis, T, B);
      s.wi = normalize3(axis * cosine + (T * cosf(phi) + B * sinf(phi)) * sine);
      s.valid = dot3(e.Ng, s.wi) > 0 && dot3(e.Ns, s.wi) > 0;
      s.lobe = e.mat->closure_type == 1 ? LOBE_DIFFUSE : LOBE_GLOSSY;
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

  // Hair: cosine-weighted sample around Ns. Not optimal for the tangent
  // cone lobe, but MIS against NEE covers for it and stratification is cheap.
  if (e.mat->hair_weight > 0.5f) {
    float u1 = rng.next();
    float u2 = rng.next();
    float r = sqrtf(u1);
    float phi = 2.0f * M_PIf * u2;
    float3 local = make_float3(r * cosf(phi), r * sinf(phi),
                               sqrtf(fmaxf(0.0f, 1.0f - u1)));
    s.wi = normalize3(e.T * local.x + e.B * local.y + e.Ns * local.z);
    s.valid = true;
    // Hair-Kajiya is mostly a forward-cone lobe + a diffuse base; bucket
    // it as glossy because the dominant contribution shouldn't deplete
    // the diffuse-bounces budget that thinks of brick / paint walls.
    s.lobe = LOBE_GLOSSY;
    return s;
  }

  // Coat weight mirrors eval_bsdf: scale by F_coat(V) so sampling budget
  // follows the actual coat intensity at this view angle (grazing >> normal).
  float coat_w_mat = e.coat_weight;
  float coat_alpha = e.coat_alpha;  // already filter-glossy-clamped
  float cF0 = ((e.coat_ior - 1.0f) / (e.coat_ior + 1.0f)) *
              ((e.coat_ior - 1.0f) / (e.coat_ior + 1.0f));
  float t_w = e.translucent_weight;
  // Match eval_bsdf_one's lobe gating exactly. Without the `pure_diff`
  // gate here the sampler picked the (eval-zeroed) specular lobe ~50% of
  // the time for a `ShaderNodeBsdfDiffuse` surface: those samples were
  // then evaluated as diffuse-only, biasing the throughput and dropping
  // the below-horizon spec directions (pdf<=0 break) — a Diffuse-node
  // white-furnace read 0.70 instead of 0.80. The sampler must see the
  // same weights as the evaluator or the mixture estimator is biased.
  bool pure_diff = (e.mat->pure_diffuse != 0);
  float w_base_diff = (1.0f - e.metallic) * (1.0f - e.transmission);
  float w_diffuse = w_base_diff * (1.0f - t_w) * (1.0f - e.sss_weight);
  float w_translucent = w_base_diff * t_w;
  float w_spec = pure_diff
                     ? 0.0f
                     : e.metallic + (1.0f - e.metallic) * (1.0f - e.transmission);
  float w_trans = pure_diff ? 0.0f : (1.0f - e.metallic) * e.transmission;
  float w_coat = pure_diff ? 0.0f : coat_w_mat * schlick_scalar(NoV, cF0);
  float total = w_diffuse + w_translucent + w_spec + w_trans + w_coat;
  if (total <= 0.0f)
    return s;
  float p_diff = w_diffuse / total;
  float p_translucent = w_translucent / total;
  float p_spec = w_spec / total;
  float p_trans = w_trans / total;

  float u = rng.next();
  float u1 = rng.next();
  float u2 = rng.next();

  if (u < p_diff) {
    // Cosine-weighted hemisphere sample
    float r = sqrtf(u1);
    float phi = 2.0f * M_PIf * u2;
    float3 local = make_float3(r * cosf(phi), r * sinf(phi),
                               sqrtf(fmaxf(0.0f, 1.0f - u1)));
    s.wi = normalize3(e.T * local.x + e.B * local.y + e.Ns * local.z);
    s.lobe = LOBE_DIFFUSE;
  } else if (u < p_diff + p_translucent) {
    // Translucent: cosine-weighted *back* hemisphere. Negate the local Z so
    // wi exits the surface on the side opposite Ns. Bucketed as
    // LOBE_DIFFUSE so it shares Cycles' diffuse-bounces budget — Cycles
    // categorises Translucent BSDF as a diffuse path-type as well.
    float r = sqrtf(u1);
    float phi = 2.0f * M_PIf * u2;
    float3 local = make_float3(r * cosf(phi), r * sinf(phi),
                               -sqrtf(fmaxf(0.0f, 1.0f - u1)));
    s.wi = normalize3(e.T * local.x + e.B * local.y + e.Ns * local.z);
    s.lobe = LOBE_DIFFUSE;
  } else if (u < p_diff + p_translucent + p_spec) {
    // GGX VNDF specular (anisotropic)
    float3 Vlocal = make_float3(dot3(wo, e.T), dot3(wo, e.B), dot3(wo, e.Ns));
    float3 Hlocal =
        sample_ggx_vndf_aniso(Vlocal, e.alpha_x, e.alpha_y, u1, u2);
    float3 H = normalize3(e.T * Hlocal.x + e.B * Hlocal.y + e.Ns * Hlocal.z);
    float VoH = dot3(wo, H);
    float3 wi = normalize3(H * (2.0f * VoH) - wo);
    if (dot3(e.Ns, wi) <= 0.0f)
      return s;
    s.wi = wi;
    // Fires when roughness has been clamped to the minimum (author asked for
    // a mirror); NEE can't find such a narrow lobe, so flag as specular and
    // let trace_path add BSDF-sampled emission directly.
    if (e.alpha <= 0.02f * 0.02f)
      s.specular = true;
    s.lobe = LOBE_GLOSSY;
  } else if (u < p_diff + p_translucent + p_spec + p_trans) {
    // Rough-dielectric transmission
    float eta = e.ior;
    float3 Vlocal = make_float3(dot3(wo, e.T), dot3(wo, e.B), dot3(wo, e.Ns));
    float3 Hlocal = sample_ggx_vndf(Vlocal, e.alpha, u1, u2);
    float3 H = normalize3(e.T * Hlocal.x + e.B * Hlocal.y + e.Ns * Hlocal.z);
    float VoH = dot3(wo, H);
    float F = fresnel_dielectric(VoH, eta);
    // Randomly pick reflect vs transmit by Fresnel.
    if (rng.next() < F) {
      float3 wi = normalize3(H * (2.0f * VoH) - wo);
      if (dot3(e.Ns, wi) <= 0.0f)
        return s;
      s.wi = wi;
    } else {
      float cosi = VoH;
      float eta_t = 1.0f / eta;
      float k = 1.0f - eta_t * eta_t * (1.0f - cosi * cosi);
      if (k < 0.0f)
        return s;
      float3 wt = -wo * eta_t + H * (eta_t * cosi - sqrtf(k));
      s.wi = normalize3(wt);
      if (dot3(e.Ns, s.wi) >= 0.0f)
        return s;
    }
    if (e.alpha <= 0.02f * 0.02f)
      s.specular = true;
    s.lobe = LOBE_TRANSMISSION;
  } else {
    // Coat: isotropic GGX VNDF on coat_alpha. Always reflects, so no
    // Fresnel reflect/transmit branch here.
    float3 Vlocal = make_float3(dot3(wo, e.T), dot3(wo, e.B), dot3(wo, e.Ns));
    float3 Hlocal = sample_ggx_vndf(Vlocal, coat_alpha, u1, u2);
    float3 H = normalize3(e.T * Hlocal.x + e.B * Hlocal.y + e.Ns * Hlocal.z);
    float VoH = dot3(wo, H);
    float3 wi = normalize3(H * (2.0f * VoH) - wo);
    if (dot3(e.Ns, wi) <= 0.0f)
      return s;
    s.wi = wi;
    if (coat_alpha <= 0.02f * 0.02f)
      s.specular = true;
    s.lobe = LOBE_GLOSSY;
  }

  s.valid = true;
  return s;
}

// Multi-closure eval: weighted sum over the leaves emitted by
// `eval_material`. Both f and pdf are linearly weighted; the combined
// pdf is the mixture pdf, which is what the path tracer needs for MIS
// against light pdfs (and what `sample_bsdf` below produces). When
// `g.n == 1` this is just `eval_bsdf_one(g.slots[0], wo, wi)`.
static __device__ BsdfEval eval_bsdf(const MaterialEvalGroup &g,
                                     float3 wo, float3 wi,
                                     int sampled_index = -1) {
  BsdfEval r;
  r.f = make_float3(0.0f, 0.0f, 0.0f);
  r.pdf = 0.0f;
  float total_weight=0.0f;
  for (int i = 0; i < g.n; ++i) {
    BsdfEval ev = eval_bsdf_one(g.slots[i], wo, wi);
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
                                         float3 wo, RNG &rng) {
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
  BsdfDirectionSample direction = sample_bsdf_direction(g.slots[picked], wo, rng);
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
#if (VIBRT_CLOSURE_TYPES & (1u << 5))
  if (direction.weighted_sample) {
    // Preserve the selected closure's sampled estimate and density.
    // Only the other mixture leaves need a separate evaluation.
    for (int i = 0; i < g.n; i++) {
      BsdfEval value;
      if (i == picked) {
        value.f = direction.sampled_value;
        value.pdf = direction.sampled_pdf;
      } else
        value = eval_bsdf_one(g.slots[i], wo, s.wi);
      float w = g.n == 1 ? 1 : g.weights[i];
      s.f = s.f + value.f * w;
      s.pdf += value.pdf * w;
    }
    s.pdf /= total_weight;
    return s;
  }
#endif
  if (s.transparent) {
    if (g.n == 1) {
      s.f = make_float3(1, 1, 1);
      s.pdf = 1.0f;
      return s;
    }
    float opacity=0.0f;
    for(int i=0;i<g.n;i++)opacity+=g.weights[i]*(1.0f-g.slots[i].alpha_blend);
    s.f=make_float3(opacity,opacity,opacity);
    s.pdf=opacity/fmaxf(total_weight,1e-20f);
    return s;
  }
  // Evaluate every leaf once, in the original accumulation order. Deferring
  // evaluation avoids both a redundant call and per-lane skips of sampled
  // leaves, which would still run the evaluator for other lanes in the warp.
  BsdfEval comb = g.n == 1 ? eval_bsdf_one(g.slots[0], wo, s.wi)
                          : eval_bsdf(g, wo, s.wi, picked);
  s.f = comb.f;
  s.pdf = comb.pdf;
  return s;
}

static __device__ __noinline__ void fill_surface_vertex(PathVertex*,HitGroupData*,unsigned int,float2,float3,float,unsigned int);

static __device__ __noinline__ float4 emission_alpha(const PathVertex &v,bool include_emission=true) {
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
    float3 em=make_f3(m->emission);float alpha=m->alpha_blend;
    if(include_emission&&m->emission_tex){const float *M=m->uv_transform;float2 uv=make_float2(M[0]*v.uv.x+M[1]*v.uv.y+M[2],M[3]*v.uv.x+M[4]*v.uv.y+M[5]);em=em*sample_rgba(m->emission_tex,m->emission_tex_w,m->emission_tex_h,m->emission_tex_channels,uv);}
    unsigned int outputs=(1u<<7)|(include_emission?((1u<<5)|(1u<<6)):0);
    if (m->shader_outputs & outputs) {
      vibrt_cycles::float4 out[VIBRT_SHADER_OUTPUT_COUNT];
      evaluate_shader(m, v, out,
                      vibrt_shader::EVAL_ALPHA |
                          (include_emission ? vibrt_shader::EVAL_EMISSION : 0));
      if(include_emission&&(m->shader_outputs&(1u<<5)))em=make_float3(out[5].x,out[5].y,out[5].z);
      if(include_emission&&(m->shader_outputs&(1u<<6)))em=em*out[6].x;
      if(m->shader_outputs&(1u<<7))alpha=fminf(fmaxf(out[7].x,0.0f),1.0f);
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
  float3 delta = center - P;
  float d2 = dot3(delta, delta), d = sqrtf(d2);
  float3 axis = d > 1e-12f ? delta / d : make_float3(0, 0, 1);
  if (radius <= 1e-6f) {
    position = center; wi = axis; distance = d;
    return d > 1e-6f ? 1.0f / d2 : 0;
  }
  float r2 = radius * radius;
  float3 T, B;
  build_frame(axis, T, B);
  float u = rng.next(), phi = 2 * M_PIf * rng.next();
  if (sphere) {
    bool outside = d2 > r2;
    float cos_max = sqrtf(fmaxf(1 - r2 / fmaxf(d2, r2), 0.0f));
    float one_minus_cos = outside ? r2 / d2 / (1 + cos_max) : 2;
    float ct = 1 - u * one_minus_cos;
    float st = sqrtf(fmaxf(1 - ct * ct, 0.0f));
    wi = normalize3(axis * ct + (T * cosf(phi) + B * sinf(phi)) * st);
    float root = sqrtf(fmaxf(r2 - d2 * (1 - ct * ct), 0.0f));
    distance = d * ct + (outside ? -root : root);
    position = P + wi * distance;
    return outside ? 2 / (d2 * (1 + cos_max)) : 4 / r2;
  }
  position = center + (T * cosf(phi) + B * sinf(phi)) * (radius * sqrtf(u));
  delta = position - P;
  distance = sqrtf(dot3(delta, delta));
  wi = delta / fmaxf(distance, 1e-12f);
  return fmaxf(dot3(axis, wi), 0.0f) / fmaxf(distance * distance, 1e-12f);
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
