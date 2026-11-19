#include "types.h"
#include "core.h"
#include <cstddef>

struct VibrtColorOp {
  unsigned int tag, mode;
  float params[8];
  const float *lut;
  unsigned long long lut_len;
};
static_assert(sizeof(VibrtColorOp) == 56, "VibrtColorOp ABI");

extern "C" void vibrt_color_batch(const VibrtColorOp *op,
                                  const float *a, size_t a_stride,
                                  const float *b, size_t b_stride,
                                  const float *factors, size_t factor_stride,
                                  size_t count, float *out) {
  using namespace vibrt_nodes;
  const float *p = op->params;
  for (size_t i = 0; i < count; ++i) {
    const float *src = a + i * a_stride;
    float3 rgb = {src[0], src[1], src[2]}, result = rgb;
    switch (op->tag) {
      case COLOR_NODE_MIX: {
        const float *other = b + i * b_stride;
        float fac = factors ? factors[i * factor_stride] : p[0];
        result = mix_blend_rgb(rgb, {other[0], other[1], other[2]}, fac, op->mode, p[1] != 0);
        break;
      }
      case COLOR_NODE_MATH:
        result = math_apply_rgb(rgb, op->mode, p[0], p[1], p[2] != 0);
        if (p[3] != 0) result = clamp_rgb(result, 0, 1);
        break;
      case COLOR_NODE_INVERT: result = invert_rgb(rgb, p[0]); break;
      case COLOR_NODE_HUE_SAT: result = hue_sat_rgb(rgb, p[0], p[1], p[2], p[3], p[4] != 0); break;
      case COLOR_NODE_RGB_CURVE: result = curve_rgb(rgb, op->lut, int(op->lut_len / 3), p[0], p[1], p[2], p[3] != 0); break;
      case COLOR_NODE_BRIGHT_CONTRAST: result = bright_contrast_rgb(rgb, p[0], p[1]); break;
      case COLOR_NODE_COLOR_RAMP: result = ramp_rgb(p[0] != 0 ? luminance(rgb) : rgb.x, op->lut, int(op->lut_len / 3)); break;
      case 11: result = gamma_rgb(rgb, p[0], p[1] != 0); break;
      case 12: result = clamp_rgb(rgb, p[0], p[1]); break;
      case 13: result = rgb_to_hsv_bl(rgb); break;
      case 14: result = hsv_to_rgb_bl(rgb); break;
      case 15: { float y = luminance(rgb); result = {y, y, y}; break; }
      case 16: {
        float value = scalar_math(rgb.x, op->mode, p[0], p[1]);
        result = {value, value, value};
        if (p[2] != 0 && std::isfinite(value)) result = clamp_rgb(result, 0, 1);
        break;
      }
    }
    out[i*3] = result.x;
    out[i*3+1] = result.y;
    out[i*3+2] = result.z;
  }
}
