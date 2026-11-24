#include "devicecode.h"
#include <optix.h>
#include "rect_light_sampling.h"

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif
#ifndef INV_PIf
#define INV_PIf 0.31830988618379067154f
#endif

extern "C" {
__constant__ LaunchParams params;
}

// ---------- RNG (PCG) ----------
struct RNG {
  unsigned int state;
  __device__ RNG(unsigned int pixel, unsigned int sample, unsigned int bounce) {
    state = (pixel * 17u + sample * 101u + bounce * 1999u) * 747796405u +
            2891336453u;
  }
  __device__ float next() {
    unsigned int old = state;
    state = old * 747796405u + 2891336453u;
    unsigned int w = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    unsigned int r = (w >> 22u) ^ w;
    return (r >> 9) * (1.0f / 8388608.0f);
  }
};

// ---------- Math ----------
static __forceinline__ __device__ float3 make_f3(const float *p) {
  return make_float3(p[0], p[1], p[2]);
}
static __forceinline__ __device__ float dot3(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
static __forceinline__ __device__ float3 normalize3(float3 v) {
  float len = sqrtf(dot3(v, v));
  return len > 1e-20f ? make_float3(v.x / len, v.y / len, v.z / len)
                      : make_float3(0, 0, 1);
}
static __forceinline__ __device__ float3 cross3(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}
static __forceinline__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static __forceinline__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static __forceinline__ __device__ float3 operator-(float3 a) {
  return make_float3(-a.x, -a.y, -a.z);
}
static __forceinline__ __device__ float3 operator*(float3 a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}
static __forceinline__ __device__ float3 operator*(float s, float3 a) {
  return a * s;
}
static __forceinline__ __device__ float3 operator*(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
static __forceinline__ __device__ float3 operator/(float3 a, float s) {
  return a * (1.0f / s);
}
static __forceinline__ __device__ float luminance(float3 c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

static __forceinline__ __device__ void build_frame(float3 n, float3 &t,
                                                   float3 &b) {
  float s = n.z >= 0.0f ? 1.0f : -1.0f;
  float a = -1.0f / (s + n.z);
  float bb = n.x * n.y * a;
  t = make_float3(1.0f + s * n.x * n.x * a, s * bb, -s * n.x);
  b = make_float3(bb, s + n.y * n.y * a, -n.y);
}

// ---------- Texture sampling (bilinear) ----------
static __device__ float3 sample_rgba(const float *data, int w, int h,
                                     int channels, float2 uv) {
  if (data == nullptr || w <= 0 || h <= 0)
    return make_float3(1, 1, 1);
  float u = uv.x - floorf(uv.x);
  float v = uv.y - floorf(uv.y);
  float fx = u * (float)w - 0.5f;
  // Blender's image.pixels (and what the exporter writes to the bin) is
  // bottom-up: buffer row 0 is the bottom of the image. Blender UV v=0 is
  // also the bottom, so no flip on the y axis.
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
  int y1 = wrap(y0 + 1, h);
  x0 = wrap(x0, w);
  y0 = wrap(y0, h);
  auto fetch = [&](int x, int y) {
    const float *p = &data[(y * w + x) * channels];
    return channels <= 2 ? make_float3(p[0], p[0], p[0])
                         : make_float3(p[0], p[1], p[2]);
  };
  float3 c00 = fetch(x0, y0);
  float3 c10 = fetch(x1, y0);
  float3 c01 = fetch(x0, y1);
  float3 c11 = fetch(x1, y1);
  float3 c0 = c00 * (1.0f - dx) + c10 * dx;
  float3 c1 = c01 * (1.0f - dx) + c11 * dx;
  return c0 * (1.0f - dy) + c1 * dy;
}

// ---------- Fresnel (Schlick) ----------
static __forceinline__ __device__ float schlick_scalar(float cos_theta,
                                                       float F0) {
  float x = 1.0f - fmaxf(cos_theta, 0.0f);
  float x2 = x * x;
  return F0 + (1.0f - F0) * x2 * x2 * x;
}
static __forceinline__ __device__ float3 schlick_rgb(float cos_theta,
                                                     float3 F0) {
  float x = 1.0f - fmaxf(cos_theta, 0.0f);
  float x2 = x * x;
  float x5 = x2 * x2 * x;
  return F0 + (make_float3(1, 1, 1) - F0) * x5;
}

// Dielectric Fresnel (exact unpolarised).
static __forceinline__ __device__ float fresnel_dielectric(float cos_i,
                                                           float eta) {
  cos_i = fminf(fmaxf(cos_i, -1.0f), 1.0f);
  if (cos_i < 0.0f) {
    eta = 1.0f / eta;
    cos_i = -cos_i;
  }
  float sin2_t = (1.0f - cos_i * cos_i) / (eta * eta);
  if (sin2_t >= 1.0f)
    return 1.0f;
  float cos_t = sqrtf(fmaxf(0.0f, 1.0f - sin2_t));
  float rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
  float rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
  return 0.5f * (rs * rs + rp * rp);
}

// ---------- GGX (isotropic) ----------
static __forceinline__ __device__ float ggx_D(float NoH, float alpha) {
  float a2 = alpha * alpha;
  float d = NoH * NoH * (a2 - 1.0f) + 1.0f;
  return a2 / fmaxf(M_PIf * d * d, 1e-12f);
}
static __forceinline__ __device__ float smith_G1(float NoX, float alpha) {
  float a2 = alpha * alpha;
  return 2.0f * NoX / (NoX + sqrtf(a2 + (1.0f - a2) * NoX * NoX));
}

// Sample GGX VNDF (Heitz 2018), V in local space with N=+Z.
static __device__ float3 sample_ggx_vndf(float3 V, float alpha, float u1,
                                         float u2) {
  float3 Vh = normalize3(make_float3(alpha * V.x, alpha * V.y, V.z));
  float len2 = Vh.x * Vh.x + Vh.y * Vh.y;
  float3 T1 = len2 > 0.0f
                  ? make_float3(-Vh.y, Vh.x, 0.0f) * (1.0f / sqrtf(len2))
                  : make_float3(1, 0, 0);
  float3 T2 = cross3(Vh, T1);
  float r = sqrtf(u1);
  float phi = 2.0f * M_PIf * u2;
  float t1 = r * cosf(phi);
  float t2 = r * sinf(phi);
  float s = 0.5f * (1.0f + Vh.z);
  t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;
  float3 Nh =
      T1 * t1 + T2 * t2 + Vh * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));
  float3 H =
      normalize3(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
  return H;
}

// PDF of GGX VNDF sample for reflection: D(H) * G1(V) * |V.H| / |V.N| / (4
// |V.H|) = D(H) * G1(V) / (4 |V.N|)
static __forceinline__ __device__ float ggx_vndf_pdf(float NoV, float NoH,
                                                     float alpha) {
  return ggx_D(NoH, alpha) * smith_G1(NoV, alpha) / fmaxf(4.0f * NoV, 1e-12f);
}

// ---------- GGX (anisotropic) ----------
// All inputs in the local tangent frame (z = Ns, x = T, y = B).
static __forceinline__ __device__ float
ggx_D_aniso(float3 Hlocal, float ax, float ay) {
  float x = Hlocal.x / fmaxf(ax, 1e-4f);
  float y = Hlocal.y / fmaxf(ay, 1e-4f);
  float z = Hlocal.z;
  float d = x * x + y * y + z * z;
  return 1.0f / fmaxf(M_PIf * ax * ay * d * d, 1e-12f);
}

static __forceinline__ __device__ float
smith_G1_aniso(float3 Vlocal, float ax, float ay) {
  float axv = ax * Vlocal.x;
  float ayv = ay * Vlocal.y;
  float Vz = fmaxf(Vlocal.z, 1e-6f);
  return 2.0f * Vz / fmaxf(Vz + sqrtf(axv * axv + ayv * ayv + Vz * Vz), 1e-12f);
}

static __device__ float3 sample_ggx_vndf_aniso(float3 V, float ax, float ay,
                                               float u1, float u2) {
  float3 Vh = normalize3(make_float3(ax * V.x, ay * V.y, V.z));
  float len2 = Vh.x * Vh.x + Vh.y * Vh.y;
  float3 T1 = len2 > 0.0f
                  ? make_float3(-Vh.y, Vh.x, 0.0f) * (1.0f / sqrtf(len2))
                  : make_float3(1, 0, 0);
  float3 T2 = cross3(Vh, T1);
  float r = sqrtf(u1);
  float phi = 2.0f * M_PIf * u2;
  float t1 = r * cosf(phi);
  float t2 = r * sinf(phi);
  float s = 0.5f * (1.0f + Vh.z);
  t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;
  float3 Nh =
      T1 * t1 + T2 * t2 + Vh * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));
  return normalize3(
      make_float3(ax * Nh.x, ay * Nh.y, fmaxf(0.0f, Nh.z)));
}

static __forceinline__ __device__ float
ggx_vndf_pdf_aniso(float3 Vlocal, float3 Hlocal, float ax, float ay) {
  if (Hlocal.z <= 0.0f || Vlocal.z <= 0.0f)
    return 0.0f;
  float D = ggx_D_aniso(Hlocal, ax, ay);
  float G1 = smith_G1_aniso(Vlocal, ax, ay);
  return D * G1 / fmaxf(4.0f * Vlocal.z, 1e-12f);
}

// ---------- GGX energy compensation (Kulla-Conty) ----------
// LUT layout matches build.rs::generate_ggx_energy_lut():
//   e_lut[ci * n + ai]  with  cos_theta = (ci+0.5)/n,  alpha = (ai+0.5)/n
//   e_avg[ai]           with  alpha     = (ai+0.5)/n
#define GGX_LUT_N 32

static __forceinline__ __device__ float ggx_lut_coord(float x) {
  // Map x ∈ [0,1] to fractional texel index for center-pixel-convention LUT.
  float t = x * (float)GGX_LUT_N - 0.5f;
  return fminf(fmaxf(t, 0.0f), (float)(GGX_LUT_N - 1));
}

static __device__ float ggx_e_lookup(float NoV, float alpha) {
  float fc = ggx_lut_coord(fminf(fmaxf(NoV, 0.0f), 1.0f));
  float fa = ggx_lut_coord(fminf(fmaxf(alpha, 0.0f), 1.0f));
  int c0 = (int)floorf(fc);
  int a0 = (int)floorf(fa);
  int c1 = min(c0 + 1, GGX_LUT_N - 1);
  int a1 = min(a0 + 1, GGX_LUT_N - 1);
  float dc = fc - (float)c0;
  float da = fa - (float)a0;
  const float *L = params.ggx_e_lut;
  float v00 = L[c0 * GGX_LUT_N + a0];
  float v10 = L[c1 * GGX_LUT_N + a0];
  float v01 = L[c0 * GGX_LUT_N + a1];
  float v11 = L[c1 * GGX_LUT_N + a1];
  float v0 = v00 * (1.0f - dc) + v10 * dc;
  float v1 = v01 * (1.0f - dc) + v11 * dc;
  return v0 * (1.0f - da) + v1 * da;
}

static __device__ float ggx_e_avg_lookup(float alpha) {
  float fa = ggx_lut_coord(fminf(fmaxf(alpha, 0.0f), 1.0f));
  int a0 = (int)floorf(fa);
  int a1 = min(a0 + 1, GGX_LUT_N - 1);
  float da = fa - (float)a0;
  const float *L = params.ggx_e_avg_lut;
  return L[a0] * (1.0f - da) + L[a1] * da;
}

// Fresnel split-sum bias term B(NoV, alpha) = ∫ (1-VoH)^5 · G·VoH/(NoV·NoH).
// With E = A + B, the dielectric spec directional reflectance is
// F0·A + B = F0·E + B·(1 − F0) — the grazing white-lift the old bare `F0·E`
// diffuse-under-spec coupling dropped.
static __device__ float ggx_b_lookup(float NoV, float alpha) {
  float fc = ggx_lut_coord(fminf(fmaxf(NoV, 0.0f), 1.0f));
  float fa = ggx_lut_coord(fminf(fmaxf(alpha, 0.0f), 1.0f));
  int c0 = (int)floorf(fc);
  int a0 = (int)floorf(fa);
  int c1 = min(c0 + 1, GGX_LUT_N - 1);
  int a1 = min(a0 + 1, GGX_LUT_N - 1);
  float dc = fc - (float)c0;
  float da = fa - (float)a0;
  const float *L = params.ggx_b_lut;
  float v00 = L[c0 * GGX_LUT_N + a0];
  float v10 = L[c1 * GGX_LUT_N + a0];
  float v01 = L[c0 * GGX_LUT_N + a1];
  float v11 = L[c1 * GGX_LUT_N + a1];
  float v0 = v00 * (1.0f - dc) + v10 * dc;
  float v1 = v01 * (1.0f - dc) + v11 * dc;
  return v0 * (1.0f - da) + v1 * da;
}

// Hemispherical-average Fresnel under Schlick approximation:
//   F_avg = ∫ F_Schlick(μ) · 2μ dμ  =  (20·F0 + 1) / 21
static __forceinline__ __device__ float f_avg_schlick(float F0) {
  return F0 * (20.0f / 21.0f) + (1.0f / 21.0f);
}
static __forceinline__ __device__ float3 f_avg_schlick(float3 F0) {
  return F0 * (20.0f / 21.0f) +
         make_float3(1.0f / 21.0f, 1.0f / 21.0f, 1.0f / 21.0f);
}

// ---------- Homogeneous volumes and phase functions ----------

// Maximum nesting depth for mesh-bounded volumes within a single ray. Four is
// generous: real scenes top out at "two overlapping volumes" (e.g. a fog cube
// inside a smoke cube). World volume sits implicitly below the stack.
#define VOL_STACK_MAX 4

struct VolumeStack {
  const Volume *entries[VOL_STACK_MAX];
  int depth;
};

static __forceinline__ __device__ const Volume *
volume_stack_top(const VolumeStack &s) {
  if (s.depth > 0)
    return s.entries[s.depth - 1];
  return params.world_volume; // may be nullptr (vacuum)
}

static __forceinline__ __device__ void
volume_stack_push(VolumeStack &s, const Volume *v) {
  if (v == nullptr)
    return;
  if (s.depth < VOL_STACK_MAX)
    s.entries[s.depth++] = v;
  // Overflow silently — losing the deepest nesting is preferable to corrupt
  // memory; the scene would already be pathological at this nesting depth.
}

static __forceinline__ __device__ void
volume_stack_pop(VolumeStack &s, const Volume *v) {
  // Pop by reference equality scanned from top. BVH order isn't strict LIFO
  // (a back-face exit can come before the matching front-face entry was
  // recorded if rays miss tessellation), so a strict pop would desync.
  for (int i = s.depth - 1; i >= 0; i--) {
    if (s.entries[i] == v) {
      for (int j = i; j < s.depth - 1; j++)
        s.entries[j] = s.entries[j + 1];
      s.depth--;
      return;
    }
  }
}

// Average per-channel σ_t — used as the scalar pdf for distance sampling.
// We then track per-channel beam transmittance separately so chromatic
// volumes don't lose colour information through the sampler.
static __forceinline__ __device__ float vol_sigma_t_avg(const Volume *v) {
  return (v->sigma_t[0] + v->sigma_t[1] + v->sigma_t[2]) * (1.0f / 3.0f);
}

static __forceinline__ __device__ float3 vol_beam_transmittance(const Volume *v,
                                                                float t) {
  return make_float3(expf(-v->sigma_t[0] * t), expf(-v->sigma_t[1] * t),
                     expf(-v->sigma_t[2] * t));
}

// Closed-form path emission integral over [0, t] for a homogeneous volume:
//   ∫_0^t exp(-σ_t s) σ_e ds = σ_e/σ_t · (1 - exp(-σ_t t))   (per channel)
// Computed deterministically — no MC noise from emissive media. Falls back
// to σ_e · t in the σ_t→0 limit (Beer-Lambert is identity, integrand = σ_e).
static __device__ float3 volume_emission(const Volume *v, unsigned int flags);
static __device__ float3 vol_path_emission(const Volume *v, float t, unsigned int flags) {
  float3 e = volume_emission(v, flags);
  if (e.x <= 0.0f && e.y <= 0.0f && e.z <= 0.0f)
    return make_float3(0, 0, 0);
  auto component = [t](float sig_t, float em) {
    if (em <= 0.0f)
      return 0.0f;
    if (sig_t < 1e-8f)
      return em * t;
    return em * (1.0f - expf(-sig_t * t)) / sig_t;
  };
  return make_float3(component(v->sigma_t[0], e.x),
                     component(v->sigma_t[1], e.y),
                     component(v->sigma_t[2], e.z));
}

// Henyey-Greenstein phase function p(cos θ) and importance sampling.
// p(μ) = (1 - g²) / (4π · (1 + g² - 2g·μ)^(3/2))
static __forceinline__ __device__ float phase_hg_eval(float g, float cos_theta) {
  float g2 = g * g;
  float denom = 1.0f + g2 - 2.0f * g * cos_theta;
  return (1.0f - g2) /
         (4.0f * M_PIf * sqrtf(fmaxf(denom, 1e-12f)) * fmaxf(denom, 1e-12f));
}

// Sample a scatter direction from HG phase, returned in the local frame
// {x = T_perp1, y = T_perp2, z = forward}. Caller transforms to world.
static __device__ float3 phase_hg_sample(float g, float u1, float u2,
                                         float &pdf) {
  float cos_theta;
  if (fabsf(g) < 1e-3f) {
    cos_theta = 1.0f - 2.0f * u1; // isotropic
  } else {
    float k = (1.0f - g * g) / (1.0f - g + 2.0f * g * u1);
    cos_theta = (1.0f + g * g - k * k) / (2.0f * g);
  }
  cos_theta = fmaxf(-1.0f, fminf(1.0f, cos_theta));
  float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
  float phi = 2.0f * M_PIf * u2;
  pdf = phase_hg_eval(g, cos_theta);
  return make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
}

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

// ---------- Visibility (shadow rays) ----------
// Ray types: 0 = radiance, 1 = shadow. SBT stride = 2.
// Shadow hit group
// is a dummy (CH disabled, no AH) so only the miss program runs, which sets
// payload[0]=1 meaning "reached infinity = light visible".
static __device__ bool shadow_visible(float3 P, float3 dir, float tmax) {
  unsigned int miss_flag = 0;
  // Mask bit 1 = "blocks shadow rays"; instances with cast_shadow=false clear
  // it (e.g. classroom's paper-lantern drum) so NEE sees through them.
  optixTrace(params.traversable, P, dir, 1e-4f, tmax - 1e-3f, 0.0f,
             OptixVisibilityMask(0x02),
             OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                 OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
             1, 2, 1, miss_flag);
  return miss_flag != 0;
}

// Same visibility query, but returns RGB beam transmittance through any
// volumes the shadow ray traverses on its way to the light. Volume-only
// boundaries are walked iteratively (CH fires on the radiance hit group,
// updates the local stack, and we re-trace from the next epsilon).
//
// Returns (0, 0, 0) on opaque occlusion. Capped at `VOL_STACK_MAX * 4`
// trace iterations to bound the worst case (alternating boundary
// crossings); past that we conservatively assume the light is unreachable
// — pathological scenes only.
static __device__ __noinline__ float4 emission_alpha(const PathVertex &v, bool include_emission);
static __device__ __noinline__ float3 shadow_transmittance(float3 P, float3 dir, float tmax,
                                              VolumeStack stack) {
  // Fast path: no volumes anywhere. Standard binary visibility — the
  // any-hit handler already lets transmissive surfaces pass through
  // (`m->transmission > 0.5f -> optixIgnoreIntersection`), so direct
  // light reaches the pool floor through clear water without paying
  // for the iterative loop on every shadow ray.
  if (params.world_volume == nullptr && stack.depth == 0) {
    return shadow_visible(P, dir, tmax) ? make_float3(1, 1, 1)
                                        : make_float3(0, 0, 0);
  }

  float3 tr = make_float3(1, 1, 1);
  float3 cur_origin = P;
  float remaining = tmax;
  const int max_iters = VOL_STACK_MAX * 4 + 4;

  for (int iter = 0; iter < max_iters; iter++) {
    if (remaining <= 1e-4f)
      return tr;

    // Trace using the radiance ray type so CH fires and we get hit info.
    PathVertex v;
    v.hit = 0;
    unsigned int hi, lo;
    pack_ptr(&v, hi, lo);
    optixTrace(params.traversable, cur_origin, dir, 1e-4f, remaining - 1e-3f,
               0.0f, OptixVisibilityMask(0x02), OPTIX_RAY_FLAG_NONE, 0, 2, 0,
               hi, lo);

    const Volume *cur_vol = volume_stack_top(stack);
    float t_seg;
    if (v.hit == 0) {
      t_seg = remaining; // ran out before any geometry
    } else {
      float3 d = v.P - cur_origin;
      t_seg = sqrtf(dot3(d, d));
    }

    // Beam transmittance through the current volume on this segment.
    if (cur_vol != nullptr) {
      tr = tr * vol_beam_transmittance(cur_vol, t_seg);
      if (luminance(tr) < 1e-6f)
        return make_float3(0, 0, 0);
    }

    if (v.hit == 0)
      return tr; // reached light

    // Surface in the way. Three cases:
    //   1. Pure volume container — walk past, push/pop the volume stack.
    //   2. Transmissive surface (Glass / Refraction / `transmission > 0`) —
    //      attenuate by (1−F)·base_color and continue past, so the sun
    //      reaches the pool floor through the water plane. Without this
    //      the water acts as an opaque occluder for NEE and the only
    //      light hitting underwater surfaces is the path-traced indirect
    //      bounce, which is dim and noisy.
    //   3. Anything else — opaque, return zero.
    if (v.mat != nullptr && v.mat->volume != nullptr &&
        v.mat->volume_only != 0) {
      bool entering = dot3(v.Ng, dir) < 0.0f;
      if (entering)
        volume_stack_push(stack, v.mat->volume);
      else
        volume_stack_pop(stack, v.mat->volume);
      // Advance just past the surface. Use ray dir (not Ng) — we want to
      // stay on the same side of the boundary we just crossed.
      float p_scale =
          fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
      float eps = fmaxf(1e-4f, p_scale * 1e-5f);
      cur_origin = v.P + dir * eps;
      remaining -= t_seg + eps;
      continue;
    }
    if (v.mat != nullptr && v.mat->shader_has_alpha) {
      v.path_flags = 2u;
      float transparent = 1.0f - emission_alpha(v, false).w;
      if (transparent > 0.0f) {
        tr = tr * transparent;
        if (luminance(tr) < 1e-6f) return make_float3(0,0,0);
        float eps = fmaxf(1e-4f, fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)),fabsf(v.P.z))*1e-5f);
        cur_origin = v.P + dir * eps;
        remaining -= t_seg + eps;
        continue;
      }
    }
    if (v.mat != nullptr && v.mat->transmission > 0.0f) {
      // Transparent shadow: keep the ray going through the surface,
      // attenuated by `(1 − F) · base_color · transmission`. We can't
      // call `eval_material` here because it's defined later in the
      // translation unit and forward-declaring it just to share the
      // (textured) base colour would pull in a lot of unrelated state;
      // the constant `base_color` is correct for water / clear-glass
      // (the common case) and only loses tint texture variation, which
      // is rarely present on transmissive shaders.
      float3 base = make_f3(v.mat->base_color);
      float cos_i = fabsf(dot3(v.Ns, dir));
      float F0_d = ((v.mat->ior - 1.0f) / (v.mat->ior + 1.0f)) *
                   ((v.mat->ior - 1.0f) / (v.mat->ior + 1.0f))
                   * (2.0f * v.mat->specular_ior_level);
      float F = schlick_scalar(cos_i, F0_d);
      float3 t_factor = base * (1.0f - F) * v.mat->transmission;
      // Surfaces with `transmission < 1` (e.g. Principled at 0.5) still
      // partially absorb — fold the missing fraction into 0 so the ray
      // does NOT pretend to also transmit the diffuse / specular share.
      tr = tr * t_factor;
      if (luminance(tr) < 1e-6f)
        return make_float3(0, 0, 0);
      float p_scale =
          fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
      float eps = fmaxf(1e-4f, p_scale * 1e-5f);
      cur_origin = v.P + dir * eps;
      remaining -= t_seg + eps;
      continue;
    }
    return make_float3(0, 0, 0); // opaque occlusion
  }
  // Exceeded the iteration cap — conservatively block.
  return make_float3(0, 0, 0);
}

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
                                      const VolumeStack &vstack, bool rect_mis) {
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
    BsdfEval b = eval_bsdf(e, wo, wi);
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
    BsdfEval b = eval_bsdf(e, wo, wi);
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
    BsdfEval b = eval_bsdf(e, wo, wi);
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
            BsdfEval b = eval_bsdf(e, wo, wi);
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
                BsdfEval b = eval_bsdf(e, wo, wi);
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
        BsdfEval b = eval_bsdf(e, wo, es.dir);
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
          BsdfEval b = eval_bsdf(e, wo, wi);
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
struct SceneQuery {
  HitGroupData *geometry;
  unsigned int hit_primitive;
  bool hit;
  float distance;
  float2 bary;
  float3 P, N, Ng;
};
static __device__ bool query_box(const QueryNode &node, float3 P, float3 D,
                                 float tmin, float tmax) {
  const float *p = &P.x, *d = &D.x;
  for (int axis = 0; axis < 3; axis++) {
    if (fabsf(d[axis]) < 1e-30f) {
      if (p[axis] < node.lo[axis] || p[axis] > node.hi[axis])
        return false;
    } else {
      float a = (node.lo[axis] - p[axis]) / d[axis],
            b = (node.hi[axis] - p[axis]) / d[axis];
      tmin = fmaxf(tmin, fminf(a, b));
      tmax = fminf(tmax, fmaxf(a, b));
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
  unsigned int stack[64];
  int sp = 0;
  stack[sp++] = 0;
  while (sp) {
    const QueryNode &node = obj.nodes[stack[--sp]];
    if (!query_box(node, P, D, tmin, q.distance))
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
  unsigned int stack[64];
  int sp = 0;
  stack[sp++] = 0;
  while (sp) {
    const QueryNode &node = params.query_nodes[stack[--sp]];
    if (!query_box(node, origin, direction, tmin, q.distance))
      continue;
    if (node.count == 0) {
      stack[sp++] = node.first;
      stack[sp++] = node.first + 1;
      continue;
    }
    for (unsigned int i = 0; i < node.count; i++) {
      const QueryObject &obj =
          params.query_objects[params.query_indices[node.first + i]];
      if (!(obj.mask & mask) ||
          (only_local && obj.geometry->instance_id != v.geometry->instance_id))
        continue;
      query_mesh(obj, v, origin, direction, tmin, skip_self, q);
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

extern "C" __global__ void __closesthit__ch() {
  fill_path_vertex(get_path_vertex());
}

extern "C" __global__ void __closesthit__shadow() {
  // Not used when TERMINATE_ON_FIRST_HIT is set — kept to satisfy SBT.
}

// Shared by radiance and shadow ray types: alpha-mask cutout, plus
// transparent-shadow pass-through for transmissive materials on shadow rays.
extern "C" __global__ void __anyhit__ah() {
  HitGroupData *hg = (HitGroupData *)optixGetSbtDataPointer();
  unsigned int prim = optixGetPrimitiveIndex();
  PrincipledGpu *m = hg->mat;
  if (hg->material_indices != nullptr && hg->num_materials > 0) {
    unsigned int mi = hg->material_indices[prim];
    if ((int)mi < hg->num_materials)
      m = hg->materials[mi];
  }
  // Shadow rays pass through transmissive surfaces (approximation of
  // Cycles' transparent shadows — no color attenuation for now).
  // Translucent thin sheets (Add(Diffuse, Translucent), sss_weight>0)
  // are NOT auto-passed: Cycles only lets ~30-50% through them, and
  // letting them be fully transparent makes the foreground pool floor
  // ~50% brighter than Cycles' reference. Cycles-style soft shadows
  // through translucent BSDFs would need a payload-accumulating
  // shadow-ray loop; for now respecting the alpha-cutout silhouette
  // (which is what's checked further below) gets us closer to
  // reference than full pass-through.
  //
  // NB: full pass-through over-delivers light entering through glass at
  // grazing angles (classroom exterior_fillLight ~2.4× over; clean
  // glass-pane test 2.58×). But BLOCKING glass shadows (the Cycles-exact
  // behaviour, delivering via refraction paths) under-delivers badly here
  // because vibrt's refraction paths under-sample at practical SPP
  // (flat_archiviz 1.00→0.92, junk_shop 1.07→0.73). The correct fix is
  // Fresnel-attenuated transparent shadows via an accumulating shadow-ray
  // loop; until then full pass-through is the better net approximation.
  bool is_shadow_ray =
      (optixGetRayFlags() & OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT) != 0;
  if (is_shadow_ray && m->transmission > 0.5f) {
    optixIgnoreIntersection();
  }
  // Cycles' Principled `Alpha < 1` mixes the BSDF with a Transparent
  // BSDF; shadow rays through the (1-alpha) portion pass straight
  // through and `alpha` of them are blocked by the opaque BSDF. We
  // approximate this stochastically: hash ray origin + direction +
  // primitive index into [0, 1) and ignore (pass through) when the
  // hash exceeds alpha_blend. Across many shadow rays through the
  // same surface the expected pass-through fraction is `(1-alpha_blend)`,
  // matching Cycles' transparent-branch fraction. PCG hash gives
  // per-ray decorrelation without needing to thread the path RNG
  // through OptiX's any-hit payload (which would require restructuring
  // every NEE call site). With the pure full-pass-through approximation
  // the camera-side window mix worked but rooms ended up ~5% too bright
  // because shadow rays through windows weren't attenuated.
  float shadow_alpha=m->shader_has_alpha?m->alpha_blend:1.0f;
  if(is_shadow_ray && m->shader_has_alpha && ((m->shader_outputs&(1u<<7))||m->secondary||m->left_subtree)) {
    PathVertex v;fill_path_vertex(&v);shadow_alpha=emission_alpha(v,false).w;
  }
  if (is_shadow_ray && shadow_alpha < 1.0f) {
    float3 ray_o = optixGetWorldRayOrigin();
    float3 ray_d = optixGetWorldRayDirection();
    unsigned int h = __float_as_uint(ray_o.x) ^ __float_as_uint(ray_o.y * 1.31f) ^
                     __float_as_uint(ray_o.z * 1.71f) ^
                     __float_as_uint(ray_d.x * 2.13f) ^
                     __float_as_uint(ray_d.y * 2.71f) ^
                     __float_as_uint(ray_d.z * 3.13f) ^
                     (prim * 747796405u + 2891336453u);
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    float u = (h >> 9) * (1.0f / 8388608.0f);
    if (u >= shadow_alpha) {
      optixIgnoreIntersection();
    }
  }
  // Pure volume-container surfaces are invisible to *binary* shadow rays
  // (the legacy `shadow_visible` path used by surface NEE on volumeless
  // scenes — TERMINATE_ON_FIRST_HIT is set there). Without this check the
  // smoke's mesh would shadow lights behind it as if it were opaque.
  //
  // We do NOT skip volume_only surfaces on radiance rays — `trace_path`
  // depends on closest-hit firing at the boundary so it can update the
  // volume stack. `shadow_transmittance` similarly uses the radiance ray
  // type (no terminate-on-first-hit) so CH runs there too.
  if (is_shadow_ray && m->volume != nullptr && m->volume_only != 0) {
    optixIgnoreIntersection();
  }
  if (m->alpha_threshold <= 0.0f || m->base_color_tex == nullptr)
    return;
  // Recompute UV from barycentrics.
  if (hg->uvs == nullptr)
    return;
  int i0 = hg->indices[prim * 3 + 0];
  int i1 = hg->indices[prim * 3 + 1];
  int i2 = hg->indices[prim * 3 + 2];
  float2 bary = optixGetTriangleBarycentrics();
  float b0 = 1.0f - bary.x - bary.y;
  float2 uv0 = make_float2(hg->uvs[i0 * 2 + 0], hg->uvs[i0 * 2 + 1]);
  float2 uv1 = make_float2(hg->uvs[i1 * 2 + 0], hg->uvs[i1 * 2 + 1]);
  float2 uv2 = make_float2(hg->uvs[i2 * 2 + 0], hg->uvs[i2 * 2 + 1]);
  float2 uv =
      make_float2(b0 * uv0.x + bary.x * uv1.x + bary.y * uv2.x,
                  b0 * uv0.y + bary.x * uv1.y + bary.y * uv2.y);
  const float *M = m->uv_transform;
  uv = make_float2(M[0] * uv.x + M[1] * uv.y + M[2],
                   M[3] * uv.x + M[4] * uv.y + M[5]);
  // Bilinear fetch of the alpha channel, wrapped.
  int w = m->base_color_tex_w;
  int h = m->base_color_tex_h;
  if (w <= 0 || h <= 0)
    return;
  float u = uv.x - floorf(uv.x);
  float v = uv.y - floorf(uv.y);
  float fx = u * (float)w - 0.5f;
  // Blender's image.pixels (and what the exporter writes to the bin) is
  // bottom-up: buffer row 0 is the bottom of the image. Blender UV v=0 is
  // also the bottom, so no flip on the y axis.
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
  int y1 = wrap(y0 + 1, h);
  x0 = wrap(x0, w);
  y0 = wrap(y0, h);
  const float *data = m->base_color_tex;
  int channels = m->base_color_tex_channels;
  auto fetch_alpha = [&](int x, int y) {
    return channels == 2 || channels == 4
               ? data[(y * w + x) * channels + channels - 1]
               : 1.0f;
  };
  float a00 = fetch_alpha(x0, y0);
  float a10 = fetch_alpha(x1, y0);
  float a01 = fetch_alpha(x0, y1);
  float a11 = fetch_alpha(x1, y1);
  float a0 = a00 * (1.0f - dx) + a10 * dx;
  float a1 = a01 * (1.0f - dx) + a11 * dx;
  float alpha = a0 * (1.0f - dy) + a1 * dy;
  if (alpha < m->alpha_threshold)
    optixIgnoreIntersection();
}

extern "C" __global__ void __miss__ms() {
  PathVertex *v = get_path_vertex();
  v->hit = 0;
}

extern "C" __global__ void __miss__shadow() { optixSetPayload_0(1u); }

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
                                                 float &transparency) {
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
  for (unsigned int bounce = 0; bounce < params.max_depth; bounce++) {
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
    float ray_tmin = (bounce == 0 && camera_skip == 0) ? first_ray_tmin : 1e-4f;
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
               2, // SBT stride (radiance and shadow)
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
        float p_scale =
            fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
        float eps = fmaxf(1e-3f, p_scale * 1e-5f);
        origin = v.P + dir * eps;
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
      if (sigma_t_avg > 1e-12f) {
        float u = rng.next();
        float t_scatter =
            -logf(fmaxf(1.0f - u, 1e-30f)) / sigma_t_avg;
        if (t_scatter < volume_length) {
          // --- Volume scatter event ---
          // Path emission [0, t_scatter] (deterministic, noise-free).
          L = L + throughput * vol_path_emission(cur_vol, t_scatter, last_path_flags);
          float3 P_scatter = origin + dir * (volume_start + t_scatter);
          float3 tr = vol_beam_transmittance(cur_vol, t_scatter);
          float pdf_t = sigma_t_avg * expf(-sigma_t_avg * t_scatter);
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
          dir = wi;
          continue;
        }
        // No scatter — path emission across the full segment, then apply
        // beam transmittance to the endpoint and divide by survival prob so
        // the no-scatter branch stays an unbiased estimator.
        L = L + throughput * vol_path_emission(cur_vol, volume_length, last_path_flags);
        float3 tr = vol_beam_transmittance(cur_vol, volume_length);
        float pdf_surv = expf(-sigma_t_avg * volume_length);
        throughput = throughput * tr / fmaxf(pdf_surv, 1e-30f);
      } else {
        // Pure emitter (σ_t = 0): only path emission, no transmittance work.
        L = L + throughput * vol_path_emission(cur_vol, volume_length, last_path_flags);
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
      float p_scale = fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
      float eps = fmaxf(1e-4f, p_scale * 1e-5f);
      origin = v.P + dir * eps;
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
    v.shader_seed = rng.state;
    v.path_depth = bounce;
    v.diffuse_depth = diffuse_bounces;
    v.glossy_depth = glossy_bounces;
    v.transmission_depth = transmission_bounces;
    v.path_flags = last_path_flags;
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
      // Mesh-light pdf evaluated at this hit. The joint pmf
      // `pmf_outer × pmf_tri` reduces to `area × lum / total_power`
      // regardless of which mesh-light owns this triangle (the
      // per-light index cancels in the product), so we don't need to
      // look up `mesh_light_idx` to recover the right pdf.
      float3 ld = v.P - prev_nee_position;
      float d2 = dot3(ld, ld);
      float lum_e = luminance(e.emission);
      // `dir` is the incoming ray direction (origin → hit). cos_light is
      // the angle between the surface normal and -dir. Two-sided emitters
      // (the default for emissive meshes registered as mesh-lights —
      // matches the NEE block above) take the absolute value.
      float cos_light = fabsf(dot3(v.Ng, dir));
      if (cos_light > 1e-6f) {
        float pmf = v.prim_area * v.light_pdf;
        float pdf_solid = d2 / fmaxf(v.prim_area * cos_light, 1e-30f);
        float pdf_light = pdf_solid * pmf;
        float w_bsdf = power_heuristic(prev_bsdf_pdf, pdf_light);
        float3 e_contrib = throughput * e.emission * w_bsdf;
        e_contrib = clamp_indirect(e_contrib, params.clamp_indirect);
        L = L + e_contrib;
      }
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
      float fraction = __saturatef(sub.sss_weight);
      bool standalone = sub.mat->closure_type == 7;
      if (sub.mat->sss_method == 0 && !standalone) fraction=0;
      float strength = standalone ? 1 : fraction*(1-sub.metallic)*(1-sub.transmission)*(1-sub.translucent_weight)*sub.alpha_blend;
      float NoV = fmaxf(dot3(sub.Ns,-dir),0);
      float reflectance = 0, coat = 0;
      if (!standalone && !sub.mat->pure_diffuse) {
        float f0=(sub.ior-1)/(sub.ior+1);f0=f0*f0*2*sub.specular_ior_level;
        reflectance=__saturatef(f0*ggx_e_lookup(NoV,sub.alpha)+(1-f0)*ggx_b_lookup(NoV,sub.alpha));
        float cf=(sub.coat_ior-1)/(sub.coat_ior+1);
        coat=sub.coat_weight*schlick_scalar(NoV,cf*cf);
        strength *= (1-reflectance)*(1-coat);
      }
      float other = standalone ? 0 : (1-fraction)*(1-sub.metallic)*(1-sub.transmission) +
          sub.metallic+sub.transmission+reflectance+coat+sub.sheen_weight+(1-sub.alpha_blend);
      float probability=strength/fmaxf(strength+other,1e-12f);
      if (strength>0 && rng.next()<probability) {
        throughput=throughput*(strength/probability);
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
    float3 nee = throughput * direct_light(e_group, v.P, wo, rng, vstack,
                                             bounce + 1 < params.max_depth);
    // bounce==0 → camera-ray NEE, clamp via Cycles' sample_clamp_direct.
    // bounce>0 → indirect-bounce NEE, clamp via sample_clamp_indirect.
    // Cycles applies clamping at exactly the same path-vertex layering.
    if (bounce == 0)
      nee = clamp_indirect(nee, params.clamp_direct);
    else
      nee = clamp_indirect(nee, params.clamp_indirect);
    L = L + nee;

    // Sample BSDF for next bounce
    BsdfSample bs = sample_bsdf(e_group, wo, rng);
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
      float eps = fmaxf(1e-4f, fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)),fabsf(v.P.z))*1e-5f);
      origin = v.P + bs.wi * eps;
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

    // Offset along geometric normal to reduce self-intersection. Scale with
    // |P| so large-coordinate scenes (e.g. archviz authored in cm) don't
    // fall below float-precision noise while the 1e-4 floor still covers
    // unit-scale scenes.
    float3 offset_n = dot3(bs.wi, v.Ng) > 0.0f ? v.Ng : -v.Ng;
    float p_scale = fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
    float eps = fmaxf(1e-4f, p_scale * 1e-5f);
    origin = v.P + offset_n * eps;
    dir = bs.wi;
  }
  return L;
}

extern "C" __global__ void __raygen__rg() {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  unsigned int pixel = idx.y * dim.x + idx.x;

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
                 OptixVisibilityMask(0x01), OPTIX_RAY_FLAG_NONE, 0, 2, 0, hi, lo);
      if (v.hit == 0)
        break;
      if (v.mat == nullptr || v.mat->volume == nullptr ||
          v.mat->volume_only == 0)
        break;
      float p_scale =
          fmaxf(fmaxf(fabsf(v.P.x), fabsf(v.P.y)), fabsf(v.P.z));
      float eps = fmaxf(1e-4f, p_scale * 1e-5f);
      origin = v.P + dir0 * eps;
      ray_tmin = 1e-4f;
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
  const unsigned int sample_end = params.sample_start + params.sample_count;
  for (unsigned int s = params.sample_start; s < sample_end; s++) {
    RNG rng(pixel, s, 0u);
    float jx = rng.next();
    float jy = rng.next();
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
      float r = sqrtf(u1);
      float phi = 2.0f * M_PIf * u2;
      float lx = r * cosf(phi) * params.cam_lens_radius;
      float ly = r * sinf(phi) * params.cam_lens_radius;
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
    accum = accum +
            trace_path(origin, dir, rng, params.cam_clip_start, transparency);
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
  accum = accum * inv;

  // y is 0 at top of image; we'll flip on CPU side for EXR.
  float *out = &params.image[pixel * 4];
  out[0] = accum.x;
  out[1] = accum.y;
  out[2] = accum.z;
  out[3] = 1.0f - __saturatef(transparency_sum * inv);
}
