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
