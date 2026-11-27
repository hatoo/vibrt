// Owen-scrambled Sobol from the pinned Cycles source. Each event has its
// own stream, so transparent/volume/SSS loops cannot consume a later bounce's dimensions.
struct RNG {
  unsigned int state, seed, sample, dimension;
  vibrt_cycles::float4 values;
  __device__ RNG(unsigned int pixel, unsigned int index, unsigned int domain)
      : seed(vibrt_cycles::hash_uint2(pixel, domain)), sample(index) {
    begin_event(0);
  }
  __device__ void begin_event(unsigned int event) {
    state = vibrt_cycles::hash_uint2(seed, event);
    dimension = 0;
  }
  __device__ float next() {
    if ((dimension & 3u) == 0)
      values = vibrt_cycles::sobol_burley_sample_4D(sample, dimension >> 2, state, 0xffffffffu);
    return values[(dimension++) & 3u];
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

static __forceinline__ __device__ void build_frame(float3 n, float3 &t,
                                                   float3 &b) {
  float s = n.z >= 0.0f ? 1.0f : -1.0f;
  float a = -1.0f / (s + n.z);
  float bb = n.x * n.y * a;
  t = make_float3(1.0f + s * n.x * n.x * a, s * bb, -s * n.x);
  b = make_float3(bb, s + n.y * n.y * a, -n.y);
}

// All packed texture reads share Cycles' interpolation and address modes.
static __device__ vibrt_cycles::float4 sample_image(const float *data, int w, int h,
                                                   int channels, float u, float v,
                                                   bool equirectangular = false) {
  vibrt_image::ImageView view{data, channels, false, equirectangular};
  vibrt_image::KernelImageInfo info{w, h, equirectangular ? vibrt_image::EXTENSION_CLIP
                                                       : vibrt_image::EXTENSION_REPEAT,
                                   vibrt_image::INTERPOLATION_LINEAR, &view};
  return vibrt_image::Interpolator::interp(info, u, v);
}
static __device__ float3 sample_rgba(const float *data, int w, int h,
                                     int channels, float2 uv) {
  if (data == nullptr || w <= 0 || h <= 0)
    return make_float3(1, 1, 1);
  auto c = sample_image(data, w, h, channels, uv.x, uv.y);
  return make_float3(c.x, c.y, c.z);
}

// Native-vector adapter for the shared Cycles GGX sampler used by SSS.
static __device__ float3 sample_ggx_vndf_aniso(float3 v, float ax, float ay, float u, float w) {
  auto h = vibrt_cycles::microfacet_ggx_sample_vndf(
      vibrt_cycles::make_float3(v.x, v.y, v.z), ax, ay, vibrt_cycles::make_float2(u, w));
  return make_float3(h.x, h.y, h.z);
}

static __forceinline__ __device__ vibrt_cycles::float3 cycles_vector(float3 v) {
  return vibrt_cycles::make_float3(v.x, v.y, v.z);
}
static __forceinline__ __device__ float3 offset_surface(float3 P, float3 N, float3 D) {
  auto shifted =
      vibrt_cycles::ray_offset(cycles_vector(P), cycles_vector(dot3(N, D) >= 0 ? N : -N));
  return make_float3(shifted.x, shifted.y, shifted.z);
}
static __device__ float3 transform_normal(const float *inverse, float3 v);
