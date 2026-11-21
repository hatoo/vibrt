// Spherical-rectangle sampling: Urena et al., "An Area-Preserving
// Parametrization for Spherical Rectangles" (2013). The stable asin form
// follows Blender's Apache-2.0 implementation, intern/cycles/kernel/light/area.h.
// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define VIBRT_RECT_INLINE static __forceinline__ __device__
#else
#include <cmath>
#define VIBRT_RECT_INLINE static inline
#endif

namespace vibrt_rect {

VIBRT_RECT_INLINE float clamp(float v, float lo, float hi) {
  return fminf(fmaxf(v, lo), hi);
}

struct Distribution {
  // Coordinates relative to the shading point, in the light's U/V frame.
  float x0, x1, y0, y1, z;
  float solid_angle, b0, b1, g2, g3;
  bool spherical;
};

VIBRT_RECT_INLINE Distribution prepare(float x0, float y0, float z,
                                       float width, float height) {
  Distribution d = {x0, x0 + width, y0, y0 + height, fabsf(z),
                    0, 0, 0, 0, 0, false};
  float scale = fmaxf(fmaxf(fabsf(x0), fabsf(y0)), fmaxf(width, height));
  if (width <= 0 || height <= 0 || d.z <= 1e-5f * scale)
    return d;
  float zz = d.z * d.z;
  float n0 = -d.y0 / sqrtf(zz + d.y0 * d.y0);
  float n1 = d.x1 / sqrtf(zz + d.x1 * d.x1);
  float n2 = d.y1 / sqrtf(zz + d.y1 * d.y1);
  float n3 = -d.x0 / sqrtf(zz + d.x0 * d.x0);
  float g0 = asinf(clamp(-n0 * n1, -1, 1));
  float g1 = asinf(clamp(-n1 * n2, -1, 1));
  d.g2 = asinf(clamp(-n2 * n3, -1, 1));
  d.g3 = asinf(clamp(-n3 * n0, -1, 1));
  d.solid_angle = -(g0 + g1 + d.g2 + d.g3);
  d.b0 = n0;
  d.b1 = n2;
  // Small/far/grazing rectangles lose precision in the spherical inverse.
  // Switch BOTH sampling and PDF evaluation to exact uniform-area sampling.
  float min_n2 = fminf(fminf(n0 * n0, n1 * n1), fminf(n2 * n2, n3 * n3));
  d.spherical = d.solid_angle > 1e-3f && min_n2 < 0.99999f;
  return d;
}

VIBRT_RECT_INLINE float pdf(const Distribution &d, float x, float y) {
  if (d.spherical)
    return 1.0f / d.solid_angle;
  float area = (d.x1 - d.x0) * (d.y1 - d.y0);
  if (area <= 0 || d.z <= 0)
    return 0;
  float r2 = x * x + y * y + d.z * d.z;
  return r2 * sqrtf(r2) / (d.z * area);
}

VIBRT_RECT_INLINE void sample(const Distribution &d, float u, float v,
                              float &x, float &y) {
  if (!d.spherical) {
    x = d.x0 + u * (d.x1 - d.x0);
    y = d.y0 + v * (d.y1 - d.y0);
    return;
  }
  float a = u * d.solid_angle + d.g2 + d.g3;
  // Homogeneous version of f=(cos(a)*b0+b1)/sin(a). Avoid the division
  // and its inf/inf at samples exactly on the edge of the rectangle.
  float numerator = cosf(a) * d.b0 + d.b1;
  float denominator = sinf(a);
  float cu = copysignf(1.0f / sqrtf(numerator * numerator +
                                   d.b0 * d.b0 * denominator * denominator),
                       numerator * denominator) * fabsf(denominator);
  cu = clamp(cu, -1, 1);
  x = clamp(cu * d.z / sqrtf(fmaxf(1 - cu * cu, 1e-20f)), d.x0, d.x1);
  float r = sqrtf(x * x + d.z * d.z);
  float h0 = d.y0 / sqrtf(r * r + d.y0 * d.y0);
  float h1 = d.y1 / sqrtf(r * r + d.y1 * d.y1);
  float h = h0 + v * (h1 - h0);
  y = clamp(h * r / sqrtf(fmaxf(1 - h * h, 1e-20f)), d.y0, d.y1);
}

} // namespace vibrt_rect
#undef VIBRT_RECT_INLINE
