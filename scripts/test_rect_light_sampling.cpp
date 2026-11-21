// CPU regression for the exact header embedded in the CUDA kernel.
// clang++ -O2 -std=c++17 scripts/test_rect_light_sampling.cpp -o target/test_rect_light_sampling.exe
#include "../vibrt/src/rect_light_sampling.h"
#include <array>
#include <cassert>
#include <cstdio>

using Moments = std::array<double, 4>;

static Moments integrand(double x, double y, double z) {
  double r = std::sqrt(x*x + y*y + z*z);
  return {1, x/r, y/r, z/r};
}

static void check(float x0, float y0, float z, float width, float height) {
  auto d = vibrt_rect::prepare(x0, y0, z, width, height);
  Moments actual{}, expected{};
  constexpr int n = 512;
  double area = double(d.x1-d.x0) * double(d.y1-d.y0);
  for (int i=0; i<n; ++i) {
    for (int j=0; j<n; ++j) {
      float u = (i+0.5f)/n, v = (j+0.5f)/n, x, y;
      vibrt_rect::sample(d, u, v, x, y);
      float pdf = vibrt_rect::pdf(d, x, y);
      assert(std::isfinite(x) && std::isfinite(y) && std::isfinite(pdf) && pdf > 0);
      assert(x >= d.x0 && x <= d.x1 && y >= d.y0 && y <= d.y1);
      auto f = integrand(x, y, d.z);
      // Independent double-precision integration in planar area measure.
      double qx = d.x0 + double(u)*(d.x1-d.x0);
      double qy = d.y0 + double(v)*(d.y1-d.y0);
      double r2 = qx*qx + qy*qy + double(d.z)*d.z;
      double jacobian = d.z / (r2*std::sqrt(r2));
      auto q = integrand(qx, qy, d.z);
      for (int k=0; k<4; ++k) {
        actual[k] += f[k]/pdf/(n*n);
        expected[k] += q[k]*jacobian*area/(n*n);
      }
    }
  }
  for (int k=0; k<4; ++k) {
    double error = std::abs(actual[k]-expected[k])/expected[0];
    if (error > 0.002) {
      std::fprintf(stderr, "moment %d: %.9g != %.9g (relative %.6g), rect %.6g %.6g %.6g %.6g %.6g\n",
                   k, actual[k], expected[k], error, x0,y0,z,width,height);
      assert(false);
    }
  }
  // RNG can return exact zero. Those samples must remain finite and bounded.
  for (float u : {0.f, 0.5f, 0.9999999f}) {
    for (float v : {0.f, 0.5f, 0.9999999f}) {
      float x, y;
      vibrt_rect::sample(d, u, v, x, y);
      assert(std::isfinite(x) && std::isfinite(y));
      assert(x >= d.x0 && x <= d.x1 && y >= d.y0 && y <= d.y1);
      assert(std::isfinite(vibrt_rect::pdf(d, x, y)));
    }
  }
  std::printf("PASS %s: %.4g %.4g %.4g %.4g %.4g\n",
              d.spherical ? "solid angle" : "area fallback", x0,y0,z,width,height);
}

int main() {
  check(-1,-1,1,2,2);
  check(-1,-1,-1,2,2);
  check(0.4f,-1.7f,0.6f,3,0.9f);
  check(-10,-0.03f,0.25f,20,0.06f);
  check(-0.02f,-8,0.4f,0.04f,16);
  check(-5,-3,0.2f,9,6);
  check(0,-1,0.8f,3,2);
  check(-3,0,0.8f,2,3);
  check(3,7,10000,0.1f,0.4f);
  check(1,1,0.00001f,2,3);
  auto coplanar = vibrt_rect::prepare(-1,-1,0,2,2);
  assert(!coplanar.spherical && vibrt_rect::pdf(coplanar, 0, 0) == 0);
  auto degenerate = vibrt_rect::prepare(1,1,1,0,2);
  assert(!degenerate.spherical && vibrt_rect::pdf(degenerate, 1, 1) == 0);
  auto near_plane = vibrt_rect::prepare(-1,-1,1e-7f,2,2);
  assert(!near_plane.spherical);
  float x, y;
  vibrt_rect::sample(near_plane, 0.5f, 0.5f, x, y);
  assert(std::isfinite(vibrt_rect::pdf(near_plane, x, y)));
  std::puts("PASS coplanar, degenerate, near-plane fallbacks");
}
