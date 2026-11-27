// CPU validation accessors for the same generated kernels used by NVRTC.
#include "cycles_nodes.h"
#include <initializer_list>
extern "C" float vibrt_ies_eval(const float *data, float phi, float theta) {
  return vibrt_ies::kernel_ies_interp(data, 0, phi, theta);
}
extern "C" int vibrt_tree_sample(const vibrt_light_tree::KernelLightTreeNode *nodes, const float *p,
                                 const float *n, float u, float *pdf) {
  return vibrt_light_tree::sample(nodes, vibrt_cycles::make_float3(p[0], p[1], p[2]),
                                  vibrt_cycles::make_float3(n[0], n[1], n[2]), u, pdf);
}
extern "C" float vibrt_tree_pdf(const vibrt_light_tree::KernelLightTreeNode *nodes, int leaf,
                                const float *p, const float *n) {
  return vibrt_light_tree::pdf(nodes, leaf, vibrt_cycles::make_float3(p[0], p[1], p[2]),
                               vibrt_cycles::make_float3(n[0], n[1], n[2]));
}

// Return the failed source line, keeping assertions active in release tests.
#define CHECK(x)                                                                                   \
  if (!(x))                                                                                        \
  return __LINE__
extern "C" int vibrt_reuse_checks() {
  using namespace vibrt_cycles;
  // Continuous environment samples must reconstruct their density away from
  // CDF discontinuities, including distributions with empty rows/columns.
  const float marginal[] = {0, 0, 0.4f, 1};
  const float conditional[] = {0, 0.5f, 1, 0, 0, 1, 0, 0.2f, 1};
  vibrt_background::Distribution env{{2, 3}, marginal, conditional};
  double inverse_pdf = 0;
  for (int i = 0; i < 16384; ++i) {
    auto u = sobol_burley_sample_2D(i, 0, 27, 0xffffffffu);
    u.x = fmaxf(u.x, 1e-7f);
    u.y = fmaxf(u.y, 1e-7f);
    float pdf;
    auto d = vibrt_background::background_map_sample(&env, u, &pdf);
    auto evaluated = vibrt_background::background_map_pdf(&env, d);
    CHECK(pdf > 0 && fabsf(pdf - evaluated) < 2e-3f * pdf);
    inverse_pdf += 1 / pdf;
  }
  // Support: half of the middle latitude band plus all of the bottom band.
  CHECK(fabs(inverse_pdf / 16384 - 2 * M_PI_F) < 0.01);
  const float pixels[] = {1, 1, 1, 0, 0, 0};
  vibrt_image::ImageView image{pixels, 3, false, true};
  vibrt_image::KernelImageInfo info{1, 2, vibrt_image::EXTENSION_CLIP, 0, &image};
  CHECK(vibrt_image::Interpolator::interp(info, 0, 0).x == 1);
  CHECK(vibrt_image::Interpolator::interp(info, 1, 1).x == 0);
  for (int blades : {0, 3, 6}) {
    KernelCamera camera{(float)blades, 0.4f, 0.5f};
    double x2 = 0, y2 = 0;
    for (int i = 0; i < 4096; ++i) {
      auto p = camera_sample_aperture(&camera, sobol_burley_sample_2D(i, 0, 8, 0xffffffffu));
      CHECK(4 * p.x * p.x + p.y * p.y <= 1.00001f);
      x2 += p.x * p.x; y2 += p.y * p.y;
    }
    CHECK(fabs(4 * x2 / y2 - 1) < 0.01);
  }
  vibrt_scene_nodes::ShaderData sd{};
  sd.N = sd.Ng = sd.normal = make_float3(0, 0, 1);
  sd.tangent = make_float3(1, 0, 0);
  sd.sign = 1;
  auto mapped = vibrt_scene_nodes::normal_map(sd, make_float3(0.8f, 0.5f, 0.9f), 1, 0, false, false);
  CHECK(len(mapped - normalize(make_float3(0.6f, 0, 0.8f))) < 1e-6f);
  vibrt_differential::differential3 derivative{make_float3(0.02f, 0, 0), make_float3(0, 0.03f, 0)};
  auto bumped = vibrt_scene_nodes::bump(sd, derivative, 0, 0.00002f, 0, 1, 0.2f, 0.001f, false, false);
  CHECK(len(bumped - normalize(make_float3(-0.2f, 0, 1))) < 1e-6f);
  float film[9 * 9] = {};
  for (int p = 0; p < 9; ++p) {
    film[p * 9] = film[p * 9 + 4] = 16;
    film[p * 9 + 8] = __uint_as_float(16);
    CHECK(vibrt_adaptive::film_adaptive_sampling_convergence_check(0, film, p % 3, p / 3, 0.05f, 0, 0, 3));
  }
  film[4 * 9 + 4] = 0;
  CHECK(!vibrt_adaptive::film_adaptive_sampling_convergence_check(0, film, 1, 1, 0.05f, 1, 0, 3));
  for (int y = 0; y < 3; ++y) vibrt_adaptive::film_adaptive_sampling_filter_x(0, film, y, 0, 3, 0, 3);
  for (int x = 0; x < 3; ++x) vibrt_adaptive::film_adaptive_sampling_filter_y(0, film, x, 0, 3, 0, 3);
  for (int p = 0; p < 9; ++p) CHECK(film[p * 9 + 7] == 0);
  vibrt_shadow::ShaderData shadow{0, make_float3(0.2f, 0.5f, 0.9f)};
  CHECK(is_zero(vibrt_shadow::surface_shader_transparency(&shadow)));
  shadow.flag = vibrt_shadow::SD_TRANSPARENT;
  CHECK(len(vibrt_shadow::surface_shader_transparency(&shadow) - shadow.closure_transparent_extinction) == 0);
  // The mixture includes the exponential survival mass at the endpoint.
  // Integrating constant incoming radiance yields 1 - exp(-sigma * length).
  vibrt_volume::Ray ray{make_float3(0), make_float3(0, 0, 1)};
  vibrt_volume::EquiangularCoefficients coeffs{make_float3(1, 0, 2), {0, 5}};
  double estimate = 0;
  for (int i = 0; i < 32768; ++i) {
    float u = (i + 0.5f) / 32768, pdf, t;
    if (u < 0.5f) {
      t = vibrt_volume::volume_equiangular_sample(&ray, coeffs, 2 * u, &pdf);
      CHECK(fabsf(pdf - vibrt_volume::volume_equiangular_pdf(&ray, coeffs, t)) < 1e-5f);
    }
    else t = -logf(2 - 2 * u) / 0.2f;
    if (t >= 5) continue;
    pdf = 0.5f * (0.2f * expf(-0.2f * t) + vibrt_volume::volume_equiangular_pdf(&ray, coeffs, t));
    estimate += 0.2f * expf(-0.2f * t) / pdf;
  }
  CHECK(fabs(estimate / 32768 - (1 - exp(-1.0))) < 0.0001);
  const float3 normal = make_float3(0, 0, 1), incoming = normalize(make_float3(0.2f, 0.3f, 1));
  // Sobol: deterministic, bounded, and exactly stratified in each dimension.
  int bins[4][32] = {};
  for (uint i = 0; i < 1024; ++i) {
    auto x = sobol_burley_sample_4D(i, 3, 1234, 0xffffffffu);
    auto repeat = sobol_burley_sample_4D(i, 3, 1234, 0xffffffffu);
    for (int c = 0; c < 4; ++c) {
      CHECK(x[c] >= 0 && x[c] < 1 && x[c] == repeat[c]);
      bins[c][int(x[c] * 32)]++;
    }
  }
  for (auto &dimension : bins)
    for (int n : dimension)
      CHECK(n == 32);
  for (float x : {-10000.0f, -1e-9f, 0.0f, 1e-9f, 10000.0f}) {
    auto p = make_float3(x, x, x), n = normalize(make_float3(1, -1, 1));
    CHECK(dot(ray_offset(p, n) - p, n) > 0);
    CHECK(dot(ray_offset(p, -n) - p, n) < 0);
  }
  // Sample/evaluate agreement for both classic Hair components and Toon lobes.
  for (int transmission = 0; transmission < 2; ++transmission) {
    HairBsdf h = {};
    h.N = normal;
    h.T = make_float3(0, 1, 0);
    h.roughness1 = 0.3f;
    h.roughness2 = 0.5f;
    h.offset = -0.1f;
    if (transmission)
      bsdf_hair_transmission_setup(&h);
    else
      bsdf_hair_reflection_setup(&h);
    auto sc = reinterpret_cast<ShaderClosure *>(&h);
    int accepted = 0;
    for (int i = 0; i < 1000; ++i) {
      auto u = sobol_burley_sample_2D(i, 0, 17, 0xffffffffu);
      float3 value = {}, direction = {};
      float pdf = 0, eval_pdf;
      float2 roughness;
      sample_classic_hair(&h, transmission, normal, incoming, u, &value, &direction, &pdf,
                          &roughness);
      if (pdf <= 0)
        continue;
      auto eval = transmission ? bsdf_hair_transmission_eval(sc, incoming, direction, &eval_pdf)
                               : bsdf_hair_reflection_eval(sc, incoming, direction, &eval_pdf);
      CHECK(fabsf(eval_pdf - pdf) < 2e-4f * fmaxf(pdf, 1));
      CHECK(fabsf(eval.x - value.x) < 2e-4f * fmaxf(pdf, 1));
      ++accepted;
    }
    CHECK(accepted > 500);
  }
  for (int glossy = 0; glossy < 2; ++glossy) {
    ToonBsdf h = {};
    h.N = normal;
    h.size = 0.4f;
    h.smooth = 0.2f;
    if (glossy)
      bsdf_glossy_toon_setup(&h);
    else
      bsdf_diffuse_toon_setup(&h);
    auto sc = reinterpret_cast<ShaderClosure *>(&h);
    for (int i = 0; i < 500; ++i) {
      auto u = sobol_burley_sample_2D(i, 0, 18, 0xffffffffu);
      float3 value = {}, direction = {};
      float pdf = 0, eval_pdf;
      if (glossy)
        bsdf_glossy_toon_sample(sc, normal, incoming, u, &value, &direction, &pdf);
      else
        bsdf_diffuse_toon_sample(sc, normal, incoming, u, &value, &direction, &pdf);
      if (pdf <= 0)
        continue;
      auto eval = glossy ? bsdf_glossy_toon_eval(sc, incoming, direction, &eval_pdf)
                         : bsdf_diffuse_toon_eval(sc, incoming, direction, &eval_pdf);
      CHECK(fabsf(eval_pdf - pdf) < 1e-4f);
      CHECK(fabsf(eval.x - value.x) < 1e-4f);
    }
  }
  // Triangle sampling uses the same density for a sampled point and a BSDF hit,
  // both near (solid angle) and far (area); barycentrics reconstruct that point.
  vibrt_triangle::Triangle triangle = {
      {make_float3(-1, -1, 0), make_float3(1, -1, 0), make_float3(0, 1, 0)}, false};
  for (float height : {0.25f, 2.0f, 100.0f}) {
    float3 p = make_float3(0, 0, height);
    int accepted = 0;
    for (int i = 0; i < 1024; ++i) {
      vibrt_triangle::LightSample ls = {};
      if (!vibrt_triangle::triangle_light_sample<false>(
              &triangle, 0, 0, sobol_burley_sample_2D(i, 0, 9, 0xffffffffu), 0, &ls, p))
        continue;
      vibrt_triangle::ShaderData sd;
      sd.P = ls.P;
      sd.wi = -ls.D;
      sd.Ng = ls.Ng;
      float pdf = vibrt_triangle::triangle_light_pdf(&triangle, &sd, ls.t);
      CHECK(fabsf(pdf - ls.pdf) < 1e-3f * fmaxf(1, ls.pdf));
      auto uv = ray_triangle_uv(p, ls.D, triangle.vertices[0], triangle.vertices[1],
                                triangle.vertices[2]);
      auto point = triangle.vertices[0] * (1 - uv.x - uv.y) + triangle.vertices[1] * uv.x +
                   triangle.vertices[2] * uv.y;
      CHECK(len(point - ls.P) < 1e-4f);
      ++accepted;
    }
    CHECK(accepted > 1010);
  }
  vibrt_triangle::LightSample ls = {};
  CHECK(!vibrt_triangle::triangle_light_sample<false>(&triangle, 0, 0, make_float2(0.3f, 0.7f), 0,
                                                      &ls, make_float3(0, 0, -1)));
  // Dwivedi's phase integrates to one and its inverse CDF agrees analytically.
  for (float alpha : {0.2f, 0.6f, 0.95f, 0.999f}) {
    float v = diffusion_length_dwivedi(alpha), log = logf((v + 1) / (v - 1));
    double integral = 0;
    for (int i = 0; i < 10000; ++i) {
      float u = (i + 0.5f) / 10000, cosine = sample_phase_dwivedi(v, log, u);
      float cdf = logf((v + 1) / (v - cosine)) / log;
      CHECK(fabsf(cdf - u) < 1e-4f);
      integral += 1.0 / eval_phase_dwivedi(v, log, cosine) / 10000;
    }
    CHECK(fabs(integral - 2) < 0.002);
    Spectrum tr;
    auto pdf = subsurface_random_walk_pdf(make_float3(1, 2, 3), 0.7f, false, &tr);
    CHECK(fabsf(pdf.y - 2 * expf(-1.4f)) < 1e-6f);
    CHECK(fabsf(tr.z - expf(-2.1f)) < 1e-6f);
  }
  return 0;
}
